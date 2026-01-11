from crsp_puller import CRSPDataPuller

def main():
    print("=" * 60)
    print("GICS SECTOR CLASSIFICATION RETRIEVAL")
    print("=" * 60)

    # Initialize the data puller
    puller = CRSPDataPuller()

    # Connect to WRDS
    if not puller.connect():
        print("\n" + "=" * 60)
        print("WRDS SETUP REQUIRED")
        print("=" * 60)
        print("\nPlease set up WRDS credentials:")
        print("1. Install package: pip install wrds pandas")
        print("2. Setup credentials: python -c 'import wrds; wrds.Connection()'")
        print("3. Enter your WRDS username and password")
        print("\nNote: WRDS access typically requires university affiliation")
        return

    # Retrieve GICS sector classification
    output_file = '/Users/amandeepsingh/Desktop/Quant codes/PairsTrading_Project/gics_sector_classification.csv'
    df = puller.get_gics_sector_classification(output_file=output_file)

    if df is not None:
        print("\n" + "=" * 60)
        print("SAMPLE DATA (first 10 rows):")
        print("=" * 60)
        print(df.head(10))

    # Close connection
    puller.close()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()