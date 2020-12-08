import copy
import json
import logging
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
from functools import partial
from operator import itemgetter

import alphashape
import cartopy.crs as ccrs
import psycopg2
import s3fs

from pyspark.sql import functions as F
from pyspark.sql.functions import udf,pandas_udf,PandasUDFType
from pyspark.sql.types import ArrayType, StringType
from shapely import wkt
from shapely.errors import TopologicalError
from shapely.geometry import box, LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union, split
from sklearn.metrics.pairwise import haversine_distances
rom pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("yarn") \
    .appName("python_udf_func") \
    .enableHiveSupport() \
    .getOrCreate()


# Enable Arrow optimization and fallback if there is no Arrow installed
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")

#postgre connector
def run_query(query, conn=None):
    if conn is None:
        conn = psycopg2.connect(
            host='server.ap-southeast-1.rds.amazonaws.com',
            user=os.environ["PG_USERNAME"],
            password=os.environ["PG_PASSWORD"],
            database='location_intelligence'
            )
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    return results

def get_node_info(raw_data):
    """Creates a node dict from raw information on node.

    Parameters:
        raw_data (list): raw information in format [node_id, latitude, longitude]

    Returns:
        node (dict): node - {
                                'id',
                                'lat',
                                'lon',
                            }
    """
    node = dict()
    node['id'] = raw_data[0]
    node['lat'] = raw_data[1] / 10000000
    node['lon'] = raw_data[2] / 10000000
    return node


def get_way_from_nodes(node_data, highway=None, access=None):
    """Creates ways from node information.

    Parameters:
        node_data (list): raw information in format [node_id, latitude, longitude]

    Returns:
        way (dict): way - {
                            'nodes',
                            'highway',
                            'access',
                          }
    """
    nodes = list(map(get_node_info, node_data))
    way = dict()
    way['nodes'] = nodes
    way['highway'] = highway
    way['access'] = access
    return way


def extract_building_ways(input_polygon, user=os.environ["PG_USERNAME"], password=os.environ["PG_PASSWORD"]):
    """Searches for building footprints intersecting with an input polygon.

    Parameters:
        input_polygon(shapely.geometry.polygon.Polygon): input polygon used to define area for search

    Returns:
        ways (list of osm ways): building ways in the search area
    """
    query = POLYGON_QUERY.format(input_polygon.wkt)
    conn = psycopg2.connect(
        host='location-intelligence-cluster.cluster-ckvce9fjaook.ap-southeast-1.rds.amazonaws.com',
        user=user,
        password=password,
        database='location_intelligence'
        )
    results = run_query(query, conn)
    ways = list()
    for elem in results:
        if elem[0] < 0:
            rel_query = REL_QUERY.format(abs(elem[0]))
            rel_results = run_query(rel_query, conn)[0][0]
            ways.extend(rel_results)
        else:
            ways.append(elem[0])
    if len(ways) > 0:
        way_query = WAY_QUERY.format(','.join(map(str, ways)))
        way_results = run_query(way_query, conn)
        nodes = list(map(itemgetter(1), way_results))
        ways = list()
        for way_nodes in nodes:
            node_query = NODE_QUERY.format(','.join(map(str, way_nodes)))
            node_results = run_query(node_query, conn)
            node_results.sort(key=lambda i: way_nodes[:-1].index(i[0]))
            nodes = list(map(get_node_info, node_results))
            way = dict()
            way['nodes'] = nodes
            ways.append(way)
    return ways


def extract_road_ways(input_polygon, closed=False, user=os.environ["PG_USERNAME"], password=os.environ["PG_PASSWORD"]):
    """Searches for roads intersecting with an input polygon.

    Parameters:
        input_polygon(shapely.geometry.polygon.Polygon): input polygon used to define area for search
        closed (bool): if true returns closed ways else returns open ways

    Returns:
        ways (list of osm ways): either closed or open ways depending on choice
    """
    query = LINE_QUERY.format(input_polygon.wkt)
    conn = psycopg2.connect(
            host='location-intelligence-cluster.cluster-ckvce9fjaook.ap-southeast-1.rds.amazonaws.com',
            user=user,
            password=password,
            database='location_intelligence'
            )
    ways = list()
    closed_ways = list()
    open_ways = list()
    results = run_query(query, conn)
    for elem in results:
        if elem[0] >= 0:
            ways.append(elem)
    if len(ways) > 0:
        way_query = WAY_QUERY.format(','.join(map(str, list(map(itemgetter(0), ways)))))
        way_results = run_query(way_query, conn)
        orig_way_ids = list(map(itemgetter(0), ways))
        for way_id, way_nodes in way_results:
            idx = orig_way_ids.index(way_id)
            highway = ways[idx][1]
            access = ways[idx][2]
            node_query = NODE_QUERY.format(','.join(map(str, way_nodes)))
            node_results = run_query(node_query, conn)
            result_nodes = list(map(itemgetter(0), node_results))
            sorted_node_results = list()
            for way_node in way_nodes:
                sorted_node_results.append(node_results[result_nodes.index(way_node)])
            curr_nodes = list()
            curr_node_results = list()
            for sorted_node in sorted_node_results:
                try:
                    prev_pos = curr_nodes.index(sorted_node[0])
                    curr_node_results.append(sorted_node)
                    closed_ways.append(get_way_from_nodes(curr_node_results[prev_pos:], highway, access))
                    if prev_pos != 0:
                        open_ways.append(get_way_from_nodes(curr_node_results[:prev_pos + 1], highway, access))
                    curr_node_results = list()
                    curr_nodes = list()
                    curr_nodes.append(sorted_node[0])
                    curr_node_results.append(sorted_node)
                except ValueError:
                    curr_nodes.append(sorted_node[0])
                    curr_node_results.append(sorted_node)
            if len(curr_nodes) > 1:
                open_ways.append(get_way_from_nodes(curr_node_results, highway, access))
    if closed:
        ways = closed_ways
    else:
        ways = open_ways
    return ways


def infer_polygons(way_data):
    """"Extracts polygon information from a way

    Parameters:
        way_data (dict): way related information returned by overpass

    Returns:
        osm_polygon (list of list of 2-element tuples): list of polygons from OSM
    """
    polygon = list()
    if len(way_data['nodes']) > 2:
        for node in way_data['nodes']:
            polygon.append((node['lon'], node['lat']))
    polygon_obj = Polygon(polygon)
    return polygon_obj


def infer_lines(way_data):
    """"Extracts line information from a way

    Parameters:
        way_data (dict): way related information returned by overpass

    Returns:
        osm_polygon (list of list of 2-element tuples): list of line strings from OSM
    """
    line = list()
    for node in way_data['nodes']:
        line.append((node['lon'], node['lat']))
    line_obj = LineString(line)
    return line_obj


def polygon_overlap(overlay_poly, base_poly):
    """Calculates how much of overlay_poly intersects with base_poly.

    Parameters:
        base_poly (shapely.geometry.polygon.Polygon): base polygon with which intersection needs to be checked
        overlay_poly (shapely.geometry.polygon.Polygon): polygon whose intersection needs to be checked with base_poly

    Returns:
        overlap (float): % of overlay_poly overlapping with base_poly
    """
    overlay_area = overlay_poly.area
    intersection_area = overlay_poly.intersection(base_poly).area
    overlap = intersection_area / overlay_area
    return overlap


def get_max_distance(coordinates):
    """Gets the maximum distance between a set of co-ordinates.

    Parameters:
        coordinates (numpy array of lat, lon): list of points

    Returns:
        maximum distance between given points
    """
    distances = haversine_distances(coordinates)
    return np.max(distances)


def prune_polygon_via_highway(input_polygon, ngram=None):
    """Corrects an input polygon through open ways.

    Algorithm:
        1. Search for open ways intersecting with the input polygon
        2. Select non-private roads (via tags) from the above data
        3. Get maximum distance between any 2 points for each such road
        4. Loop across roads: If maximum distance b/w 2 points of road > maximum distance b/w 2 points of input polygon:
            a. Select the sub-polygon with largest area from among the multiple polygon that this road divides the input
               polygon

    Parameters:
        input_polygon (shapely.geometry.polygon.Polygon): input polygon
        ngram (string): Optional name of the ngram for input polygon

    Returns:
        input_polygon (shapely.geometry.polygon.Polygon): corrected input polygon
        corrected (bool): Boolean denoting if a correction was performed on input polygon
    """
    corrected = False
    lats = input_polygon.exterior.coords.xy[1]
    lngs = input_polygon.exterior.coords.xy[0]
    points = np.vstack((lats, lngs)).transpose()
    max_input_distance = get_max_distance(points)
    ways = extract_road_ways(input_polygon, closed=False)
    line_strings = list(map(infer_lines, ways))
    for idx, line_string in enumerate(line_strings):
        if ways[idx]['highway'] not in PRIVATE_ROAD_PROXIES:
            lats = line_string.coords.xy[1]
            lngs = line_string.coords.xy[0]
            points = np.vstack((lats, lngs)).transpose()
            max_distance = get_max_distance(points)
            # We will mark roads as Public roads if their length is > 2 the maximum distance between any 2 points of
            # polygon at this stage and the tags are not in private road tags
            if max_distance > 2 * max_input_distance or ways[idx]['highway'] in PUBLIC_ROAD_PROXIES:
                split_inputs = split(input_polygon, line_string)
                if not split_inputs.is_empty:
                    selected_area = 0
                    selected_polygon = 0
                    # Selecting largest area polygon
                    for sub_idx, polygon in enumerate(split_inputs):
                        curr_area = polygon.area
                        if curr_area > selected_area:
                            selected_polygon = sub_idx
                            selected_area = curr_area
                    if selected_area < input_polygon.area:
                        corrected = True
                    input_polygon = split_inputs[selected_polygon]
    return input_polygon, corrected


def correct_polygon_via_highway(input_polygon, poly_idx=-1):
    """Corrects an input polygon through closed ways.

    Algorithm:
        1. Search for closed ways intersecting with the input polygon and convert these roads to polygons
        2. Loop across polygons:
            a. if the polygon contains input polygon:
                i. if the polygon's highway or access tag indicate a private road:
                    expand the input polygon to this polygon
                ii. if the polygon's highway or access ta do not indicate it to be a public road and area < (1.5*input):
                    expand the input polygon to this polygon
            b. if the input polygon contains the polygon: do nothing
            c. if the input polygon intersects with the polygon:
                i. if the polygon's highway or access tag indicate a private road and >20% polygon overlaps with input
                   polygon: input polygon is the union of itself with this polygon
                ii. if the polygon's highway or access indicate it to be a Public road: input polygon is the larger of
                   the two sub polygon formed out of input polygon due to intersection with this polygon

    Parameters:
        input_polygon (shapely.geometry.polygon.Polygon): input polygon
        ngram (string): Optional name of the ngram for input polygon

    Returns:
        input_polygon (shapely.geometry.polygon.Polygon): corrected input polygon
        corrected (bool): Boolean denoting if a correction was performed on input polygon
    """
    from shapely.geometry.polygon import Polygon
    from shapely.ops import unary_union, split
    
    corrected = False
    ways = extract_road_ways(input_polygon, closed=True)
    polygons = list(map(infer_polygons, ways))
    private_road_proxies = copy.deepcopy(PRIVATE_ROAD_PROXIES)
    private_road_proxies.add('residential')
    # TODO: The order in which we consider these polygons will impact the final result. Do we account for this?
    for idx, polygon in enumerate(polygons):
        if polygon.contains(input_polygon):
            # The closed path contains the input polygon completely. We will extend the boundary of input polygon only
            # if the current polygon's way is marked as private
            if ways[idx]['highway'] in private_road_proxies or ways[idx]['access'] == 'private':
                input_polygon = polygon
                corrected = True
            elif polygon.area < 1.5 * input_polygon.area and ways[idx]['highway'] not in PUBLIC_ROAD_PROXIES \
                    and ways[idx]['access'] != 'public':
                input_polygon = polygon
                corrected = True
        elif input_polygon.contains(polygon):
            # The closed path is fully inside the input polygon. Ignore this path
            continue
        elif polygon.intersects(input_polygon):
            # The path intersects with the input polygon. If >20% of area of path is inside the input polygon and the
            # path is a private path, then we extend the input polygon
            if (ways[idx]['highway'] in private_road_proxies or ways[idx]['access'] == 'private') \
                    and polygon_overlap(polygon, input_polygon) > .2:
                input_polygon = unary_union([input_polygon, polygon])
                corrected = True
            # If the path is a public path, then we include only the larger portion
            elif ways[idx]['highway'] in PUBLIC_ROAD_PROXIES or ways[idx]['access'] == 'public':
                # Consider the intersection
                if polygon_overlap(input_polygon, polygon) > .5:
                    input_polygon = input_polygon.intersection(polygon)
                    corrected = True
                #  Consider the area outside intersection
                else:
                    input_polygon = input_polygon.difference(polygon)
                    corrected = True
    return input_polygon, corrected


def get_boundary_polygon(polygons):
    """Tries to create a tight boundary around given polygons

    Parameters:
        polygons (list of shapely.geometry.polygon.Polygon): input polygons for generating the boundary
        buffer (int): buffer in meter to be considered around the alphashape
    Returns:
        boundary (shapely.geometry.polygon.Polygon): boundary around given polygons
    """
    
    gpd_polygons = gpd.GeoSeries(polygons, crs='epsg:4326')
    gpd_proj = gpd_polygons.to_crs(ccrs.AlbersEqualArea().proj4_init)
    proj_polygons = list(gpd_proj.geometry)
    proj_points = np.zeros((0, 2))
    for proj_polygon in proj_polygons:
        proj_points = np.vstack((proj_points, np.array(proj_polygon.exterior)))
    alpha_shape_proj = alphashape.alphashape(proj_points)
    alpha_polygons = gpd.GeoSeries(alpha_shape_proj, crs=ccrs.AlbersEqualArea().proj4_init)
    alpha_shape = alpha_polygons.to_crs(crs='epsg:4326')[0]
    return alpha_shape


def get_buffer(polygon, poly_buffer=10.0):
    """Outputs a geofence around a given polygon

    Parameters:
        polygon (shapely.geometry.polygon.Polygon): input polygon
        poly_buffer (float): buffer in meters to be considered around the given polygon

    Returns:
        buffer_polygon (shapely.geometry.polygon.Polygon): geofence around the given polygon"""
   
    if polygon is not None:
        gpd_polygon = gpd.GeoSeries(polygon, crs='epsg:4326')
        gpd_polygon = gpd_polygon.to_crs(crs='epsg:3857').buffer(poly_buffer)
        buffer_polygon = gpd_polygon.to_crs(crs='epsg:4326')[0]
    else:
        buffer_polygon = None
    return buffer_polygon



def get_road_correction(input_polygon, poly_buffer, ignore_buffer=True):
    """Corrects an input polygon through roads.

    Parameters:
        input_polygon (shapely.polygon.Polygon): input polygon to be corrected
        poly_buffer (float): buffer (in meters) to be used around the input polygon to search for roads
        ignore_buffer (bool): If true will revert the buffer if no correction is performed

    Returns:
        input_polygon (shapely.polygon.Polygon): corrected input polygon
    """

    temp_input_polygon = input_polygon
    input_polygon = get_buffer(input_polygon, poly_buffer)
    if not input_polygon.geom_type == 'MultiPolygon':
        temp_polygon, corrected = correct_polygon_via_highway(input_polygon)
        if corrected:
            input_polygon = temp_polygon
        elif ignore_buffer:
            input_polygon = temp_input_polygon
        if not input_polygon.geom_type == 'MultiPolygon':
            temp_polygon, corrected = prune_polygon_via_highway(input_polygon)
            if corrected:
                input_polygon = temp_polygon
    elif ignore_buffer:
        input_polygon = temp_input_polygon
    return input_polygon

def get_osm_polygon(input_polygon, ngram=None, overlap_thresh=.02):
    """Constructs a PoI from OSM building footprints intersecting with the given polygon.

    Parameters:
        input_polygon ():
        ngram (string): Name of the ngram for input polygon
        overlap_thresh (float): minimum overlap threshold with input polygon for OSM building to be considered

    Returns:
    """

    from shapely.ops import unary_union, split
    
    input_polygon = wkt.loads(input_polygon)
    input_polygon, _ = prune_polygon_via_highway(input_polygon, ngram)
    osm_ways = extract_building_ways(input_polygon.minimum_rotated_rectangle)
    osm_way = False
    if len(osm_ways) > 0:
        intermediate_polygons = list(map(infer_polygons, osm_ways))
        overlaps = np.asarray(list(map(partial(polygon_overlap, base_poly=input_polygon), intermediate_polygons)))
        indexes = range(len(intermediate_polygons))
        selected_indices = np.intersect1d(np.where(overlaps > overlap_thresh), indexes)
        if len(selected_indices) == 1:
            selected_polygons = [intermediate_polygons[int(selected_indices[0])]]
            osm_way = True
        elif len(selected_indices) > 0:
            selected_polygons = list(itemgetter(*selected_indices)(intermediate_polygons))
            osm_way = True
        else:
            selected_polygons = [input_polygon]
    else:
        selected_polygons = [input_polygon]

    osm_poi = input_polygon
    temp_polygons = copy.deepcopy(selected_polygons)
    temp_polygons.append(osm_poi)
    osm_poi_corrected = unary_union(temp_polygons)
    osm_poi_corrected = Polygon(osm_poi_corrected.exterior)
    # print(f'osm corrected polygon: {osm_poi_corrected}')
    osm_poi_corrected = get_road_correction(osm_poi_corrected, 10, ignore_buffer=True)
    # print(f'osm road corrected polygon: {osm_poi_corrected}')
    if osm_way:
        buffered_polygon_5m_corrected = get_road_correction(osm_poi_corrected, 5, ignore_buffer=False)
    else:
        buffered_polygon_5m_corrected = get_road_correction(osm_poi_corrected, 1, ignore_buffer=False)
    return str(buffered_polygon_5m_corrected.wkt)

def read_input_polygons(file_path, column_name, ops_column_name, ngram_col_name, city_id=None):
    """Parses a given csv file though geopandas and reads the individual polygons from given column

    Parameters:
        file_path (str): path to data file
        column_name (str): name of the column in above file containing the ngram polygons. This column should have
            values of type - POLYGON ((77.32772064208984 28.54449081420898, 77.32770538330078 28.54449272155762, ...))
        ops_column_name (str): name of the column in above file containing the ops corrected polygons. This column
            should have values of type -
            POLYGON ((77.32772064208984 28.54449081420898, 77.32770538330078 28.54449272155762, ...))
        ngram_col_name (str): name of the column containing ngram identifier
        city_id (int): extract data corresponding to only this city. Default = None extracts for all.

    Returns:
        ngram_polygons (list of shapely.geometry.polygon.Polygon): ngram polygons
        ngrams (list of str): ngrams representing each polygon
    """
    data = pd.read_csv(file_path)
    if city_id is not None:
        data = data[data['city_id'] == city_id]
    data[column_name] = data[column_name].apply(wkt.loads)
    # Doing the null check as there may be missing values in this column Ex. No actual PoI in the concerned area etc.
    data[ops_column_name] = data[ops_column_name].apply(lambda x: wkt.loads(x) if(str(x) != 'nan') else x)
    gdf = gpd.GeoDataFrame(data[column_name], crs='epsg:4326')
    ngram_polygons = gdf[column_name].tolist()
    gdf_ops = gpd.GeoDataFrame(data[ops_column_name], crs='epsg:4326')
    ngrams = data[ngram_col_name].tolist()
    return ngram_polygons, ngrams
    
 
osm_polygons_spark = udf(get_osm_polygon, returnType=StringType())

input_file = 'PoI_Validation_data_Overall.csv'
input_polygons, ngrams = read_input_polygons(input_file, 'geometry', 'Updated Polygon', 'ngram')
wkt_polygons = [input_polygon.wkt for input_polygon in input_polygons]

columns = ['input_polygon', 'ngram']
data = [(wkt_polygons[idx], ngrams[idx]) for idx in range(len(ngrams))]
df = spark.createDataFrame(data=data,schema=columns)

df = df.withColumn("osm_polygon", osm_polygons_spark(df['input_polygon']))   

output_file = 's3://Dataset/daily/2020-11-07/osm_polygons.csv'

df.coalesce(1).write.mode("overwrite").option("header","true").csv(output_file)