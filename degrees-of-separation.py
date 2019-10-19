#Boilerplate stuff:
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[8]").setAppName("DegreesOfSeparation")
sc = SparkContext(conf=conf)

# The characters we wish to find the degree of separation between:
startCharacterID = 5306  # SpiderMan
targetCharacterID = 14  # ADAM 3,031 (who?)

# Our accumulator, used to signal when we find the target character during
#  our BFS traversal.
# Note: An accumulator is a counter shared by all nodes in the cluster
hitCounter = sc.accumulator(0)

# convertToBFS sets up initial conditions for the graph
# convertToBFS converts the input data to node structures that we'll use and update
#
# Marvel-Graph.txt data format:
# First value = Marvel hero ID
# Remaining values = hero IDs the first hero ID has appeared with in comic books
#
# Note: The entries for a specific Hero ID can span multiple lines
#
# Example:
# 5983 1165 3836 4361 1282 716 4289 4646 6300 5084 2397 4454 1913 5861 5485
def convertToBFS(line):
    # Get all fields on the line
    fields = line.split()

    # Get the heroID for the node
    heroID = int(fields[0])

    # Collect a Python list of all connections (all fields after the first)
    connections = []
    for connection in fields[1:]:
        connections.append(int(connection))

    # Default color for every node that isn't the node for the starting character is 'WHITE'
    color = 'WHITE'
    # Default distance for every node that isn't the node for the starting character is infinity == 9999
    distance = 9999

    # If it is the starting character - the default color is 'GRAY' and the distance is 0
    if (heroID == startCharacterID):
        color = 'GRAY'
        distance = 0

    # Return the default key == heroID, value = connections, distance, color) pair representing the node
    return (heroID, (connections, distance, color))


# Import the file and construct the initial nodes using the convertToBFS function
def createStartingRdd():
    inputFile = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Marvel-Graph.txt")
    return inputFile.map(convertToBFS)


# Define the logic to be iteratively executed on each node
def bfsMap(node):
    # Extract the values from the node's data: (heroID, (connections, distance, color))
    characterID = node[0]  # heroID
    data = node[1]  # (connections, distance, color)
    connections = data[0]  # connections
    distance = data[1]  # distance
    color = data[2]  # color

    # Create a list to store node updates
    results = []

    # If this node needs to be expanded...
    # Note: initially this is just the starting character's node
    if (color == 'GRAY'):
        # For each connection to the gray node...
        for connection in connections:
            # Set the values to overwrite the connection with
            newCharacterID = connection
            newDistance = distance + 1
            newColor = 'GRAY'

            # Check if it's the target character
            if (targetCharacterID == connection):
                hitCounter.add(1)

            # Create the updated node
            newEntry = (newCharacterID, ([], newDistance, newColor))

            # Append the updated node to the list of node updates (== results)
            results.append(newEntry)

        # Once each connection is processed, this node has been fully processed, so change the color to black
        color = 'BLACK'

    # Append the color update for this node (the input node) to the list of results == node updates
    results.append(
        (characterID, (connections, distance, color))
    )

    # Return the list of results containing the new/updated nodes
    return results


def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]

    distance = 9999
    color = color1
    edges = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK')):
        color = color2

    if (color1 == 'GRAY' and color2 == 'BLACK'):
        color = color2

    if (color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK')):
        color = color1

    if (color2 == 'GRAY' and color1 == 'BLACK'):
        color = color1

    return (edges, distance, color)


#Main program here:
iterationRdd = createStartingRdd()

for iteration in range(0, 10):
    print("Running BFS iteration# " + str(iteration+1))

    # Create new vertices as needed to darken or reduce distances in the
    # reduce stage. If we encounter the node we're looking for as a GRAY
    # node, increment our accumulator to signal that we're done.
    mapped = iterationRdd.flatMap(bfsMap)

    # Note that mapped.count() action here forces the RDD to be evaluated, and
    # that's the only reason our accumulator is actually updated.
    print("Processing " + str(mapped.count()) + " values.")

    if (hitCounter.value > 0):
        print("Hit the target character! From " + str(hitCounter.value) \
            + " different direction(s).")
        break

    # Reducer combines data for each character ID, preserving the darkest
    # color and shortest path.
    iterationRdd = mapped.reduceByKey(bfsReduce)
