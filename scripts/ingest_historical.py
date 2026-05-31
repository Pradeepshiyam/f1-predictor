import argparse
import os
from f1_predictor.ingestion.fastf1_client import FastF1Client
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Ingest historical F1 data")
    parser.add_argument("--start_year", type=int, default=2021, help="Year to start ingestion from")
    parser.add_argument("--end_year", type=int, default=2023, help="Year to end ingestion at")
    parser.add_argument("--output_dir", type=str, default="data/raw/historical", help="Directory to save data")
    
    args = parser.parse_args()
    
    client = FastF1Client()
    output_path = Path("data/bronze/historical")
    output_path.mkdir(parents=True, exist_ok=True)
    
    for year in range(args.start_year, args.end_year + 1):
        print(f"--- Ingesting Season {year} ---")
        schedule = client.fetch_season_data(year)
        
        for _, event in schedule.iterrows():
            gp_name = event['EventName']
            print(f"Processing {gp_name}...")
            
            try:
                # Race Results
                results = client.get_race_results(year, gp_name)
                if results is not None and not results.empty:
                    results_file = output_path / f"{year}_{gp_name.replace(' ', '_')}_results.csv"
                    results.to_csv(results_file, index=False)
                    print(f"Successfully saved race results for {gp_name}")
                else:
                    print(f"Skipping {gp_name} - No race results available yet.")
                
                # Qualifying Results
                qual_results = client.get_qualifying_results(year, gp_name)
                if qual_results is not None and not qual_results.empty:
                    qual_file = output_path / f"{year}_{gp_name.replace(' ', '_')}_qualifying.csv"
                    qual_results.to_csv(qual_file, index=False)
                    print(f"Successfully saved qualifying for {gp_name}")
                
            except Exception as e:
                print(f"Notice: {gp_name} data not yet available on F1 servers: {e}")

if __name__ == "__main__":
    main()
