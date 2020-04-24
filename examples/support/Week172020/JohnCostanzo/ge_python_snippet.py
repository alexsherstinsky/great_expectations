    df = spark.read.parquet(hdfs_file)
    df.show(20, False)

    context = ge.DataContext()
    batch_kwargs = {
        "datasource": "leo_datasource",
        "dataset": df.limit(1000),
    }

    # profile the dataset and create or overwrite the given expectation suite name
    profile_results = context.profile_data_asset(
        datasource_name="leo_datasource",
        expectation_suite_name=category,
        batch_kwargs=batch_kwargs,
    )

    if profile_results["success"]:
        logger.info("Profiling completed")
        context.build_data_docs()
    else:
        logger.error("Profiling failed")
