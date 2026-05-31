import os
import sys

# Auto-configure environment for Spark on Windows
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin") + \
                      os.pathsep + os.path.join(os.environ["HADOOP_HOME"], "bin")

from pyspark.sql import SparkSession, functions as F, Window

def create_spark_session():
    return SparkSession.builder \
        .appName("F1DataProcessor") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def process_historical_data(spark, bronze_path, silver_path, gold_path):
    """Clean data into Silver and transform into Gold features."""
    print(f"Loading raw data from {bronze_path}...")
    # Read from Bronze
    df = spark.read.option("header", "True").csv(f"{bronze_path}/*.csv")
    
    # --- SILVER LAYER: Cleaning ---
    df = df.withColumn("ClassifiedPosition", 
                       F.when(F.col("ClassifiedPosition").rlike("^[0-9]"), F.col("ClassifiedPosition"))
                       .otherwise("22"))
    df = df.withColumn("GridPosition", 
                       F.when(F.col("GridPosition").rlike("^[0-9]"), F.col("GridPosition"))
                       .otherwise("22"))

    df = df.withColumn("ClassifiedPosition", F.col("ClassifiedPosition").cast("float").cast("int")) \
           .withColumn("GridPosition", F.col("GridPosition").cast("float").cast("int")) \
           .withColumn("Points", F.col("Points").cast("float")) \
           .withColumn("Year", F.split(F.input_file_name(), "_").getItem(0))
    
    print(f"Saving cleaned data to {silver_path}...")
    df.write.mode("overwrite").parquet(silver_path)

    # --- GOLD LAYER: Feature Engineering ---
    cleaned_df = spark.read.parquet(silver_path)
    window_spec = Window.partitionBy("FullName").orderBy("Year")
    
    # Calculate Rolling Form (Average Finish of last 5 races)
    gold_df = cleaned_df.withColumn("AvgFinishLast5", 
                                  F.avg("ClassifiedPosition").over(window_spec.rowsBetween(-5, -1)))
    
    # Calculate Team Form (Total points for team in recent races)
    team_window = Window.partitionBy("TeamName").orderBy("Year")
    gold_df = gold_df.withColumn("TeamForm", 
                               F.sum("Points").over(team_window.rowsBetween(-5, -1)))
    
    print(f"Saving final features to {gold_path}...")
    gold_df.write.mode("overwrite").parquet(gold_path)

if __name__ == "__main__":
    bronze = "data/bronze/historical"
    silver = "data/silver/historical"
    gold = "data/gold/features.parquet"
    
    # Create folders if they don't exist
    for p in [bronze, silver, "data/gold"]:
        os.makedirs(p, exist_ok=True)
        
    spark = create_spark_session()
    process_historical_data(spark, bronze, silver, gold)
    spark.stop()
