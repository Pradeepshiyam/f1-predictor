from pyspark.sql import DataFrame
from pyspark.sql.functions import count, desc, sum, when


def build_driver_form_features(results_df: DataFrame) -> DataFrame:
    """Simple baseline feature table using counts of podiums/wins by driver."""
    return (
        results_df.groupBy("driver_number", "full_name", "team_name")
        .agg(
            count("position").alias("starts"),
            sum(when(results_df["position"] == 1, 1).otherwise(0)).alias("wins"),
            sum(when(results_df["position"] <= 3, 1).otherwise(0)).alias("podiums"),
            sum("driver_points").alias("driver_points"),
        )
        .orderBy(desc("wins"), desc("podiums"), desc("driver_points"))
    )
