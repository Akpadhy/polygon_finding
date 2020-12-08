#polygon_finding
earches for building footprints intersecting with an input polygon
Searches for roads intersecting with an input polygon
Gets the maximum distance between a set of co-ordinates.


Algorithm:
        1. Search for open ways intersecting with the input polygon
        2. Select non-private roads (via tags) from the above data
        3. Get maximum distance between any 2 points for each such road
        4. Loop across roads: If maximum distance b/w 2 points of road > maximum distance b/w 2 points of input polygon:
            a. Select the sub-polygon with largest area from among the multiple polygon that this road divides the input
               polygon


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