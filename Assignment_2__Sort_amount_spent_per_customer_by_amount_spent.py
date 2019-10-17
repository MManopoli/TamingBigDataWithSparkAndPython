# Objective is to total amount spent by each customer sorted by total spending
#
# Data:
# Customer ID, Item ID, Amount Spent
# 44,8602,37.19
# 35,5368,65.89
# 2,3391,40.64
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("CustomerSpendingSorted")
sc = SparkContext(conf=conf)

lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/customer-orders.csv")
def parseLine(line):
    fields = line.split(',')
    customerID = int(fields[0])
    orderCost = float(fields[2])
    return (customerID, orderCost)


parsedLines = lines.map(parseLine)
aggCustomerSpending = parsedLines.reduceByKey(lambda x, y: x + y)
aggCustomerSpendingSorted = aggCustomerSpending.map(lambda x: (x[1], x[0])).sortByKey()
customer_spending: list = aggCustomerSpendingSorted.collect()

# sorted_customer_spending = customer_spending.sort(key=lambda x: x[1]) - the python way

for customer in customer_spending:
    # print("Customer {0} spent ${1:.2f} in total".format(customer[0], customer[1])) - If you sorted in Python
    print("Customer {0} spent ${1:.2f} in total".format(customer[1], customer[0]))
