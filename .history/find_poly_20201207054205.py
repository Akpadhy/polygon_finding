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

