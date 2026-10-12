
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PurchasesAnalysis")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

base_path = os.path.dirname(os.path.abspath(__file__))

users_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(os.path.join(base_path, "users.csv"))
)

purchases_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(os.path.join(base_path, "purchases.csv"))
)

products_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(os.path.join(base_path, "products.csv"))
)

print("Users")
users_df.show(5, truncate=False)

print("Purchases")
purchases_df.show(5, truncate=False)

print("Products")
products_df.show(5, truncate=False)

users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()

print("Кількість користувачів:", users_df.count())
print("Кількість покупок:", purchases_df.count())
print("Кількість продуктів:", products_df.count())

purchases_with_details = (
    purchases_df
    .join(users_df, on="user_id", how="inner")
    .join(products_df, on="product_id", how="inner")
    .withColumn("age", F.col("age").cast("integer"))
    .withColumn("quantity", F.col("quantity").cast("double"))
    .withColumn("price", F.col("price").cast("double"))
    .withColumn(
        "total_amount",
        F.col("quantity") * F.col("price")
    )
)

print("Покупки з інформацією про користувачів і товари")
purchases_with_details.show(10, truncate=False)


category_totals = (
    purchases_with_details
    .groupBy("category")
    .agg(
        F.sum("total_amount").alias("total_spending")
    )
    .orderBy(F.desc("total_spending"))
)

print("1. Загальні витрати за категоріями")
category_totals.show(truncate=False)


young_users_purchases = purchases_with_details.filter(
    (F.col("age") >= 18) & (F.col("age") <= 25)
)

young_category_totals = (
    young_users_purchases
    .groupBy("category")
    .agg(
        F.sum("total_amount").alias("young_spending")
    )
    .orderBy(F.desc("young_spending"))
)

print("2. Витрати користувачів 18–25 років")
young_category_totals.show(truncate=False)


total_young_spending = (
    young_users_purchases
    .agg(F.sum("total_amount").alias("total"))
    .first()["total"]
)

if total_young_spending is not None and total_young_spending != 0:

    category_percentages = (
        young_category_totals
        .withColumn(
            "spending_percentage",
            F.round(
                F.col("young_spending")
                / F.lit(total_young_spending)
                * 100,
                2
            )
        )
        .orderBy(F.desc("spending_percentage"))
    )

    print("3. Частка витрат за категоріями, %")
    category_percentages.show(truncate=False)

    top_3_categories = category_percentages.limit(3)

    print("4. Топ-3 категорії за витратами віку 18–25")
    top_3_categories.show(truncate=False)

else:
    print("Немає покупок користувачів віком 18–25 років.")

spark.stop()
