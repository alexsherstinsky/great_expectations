def validate_source_data(table, suite_name, **kwargs):
    """ Validate source table with Great Expectation Suite """
    schema = kwargs.get("templates_dict").get("schema")
    ge_base = os.path.join(os.path.dirname(__file__), "great_expectations")
    context = ge.data_context.DataContext(ge_base)

    # data sources are queryable
    data_source = SqlAlchemyDatasource(
        name="redshift",
        credentials={
            "connection_string": os.getenv("AIRFLOW_CONN_DATA_WAREHOUSE"),
            "query": {"sslmode": "prefer"},
        },
        generators={"default": {"class_name": "TableBatchKwargsGenerator"}},
    )

    # batch is a slice of data from the source
    batch_kwargs = data_source.build_batch_kwargs("default", f"{schema}.{table}")
    batch = data_source.get_batch(batch_kwargs=batch_kwargs)
    dataset = SqlAlchemyDataset(
        expectation_suite=context.get_expectation_suite(suite_name),
        expectation_suite_name=suite_name,
        **batch.data.get_init_kwargs(),
    )

    # made up run id to use for this validation
    ge_run_id = (
        f"airflow-{kwargs['dag_run'].dag_id}-"
        f"{kwargs['dag_run'].run_id}-{str(kwargs['dag_run'].start_date)}"
    )

    # run validation and run through the action list to write results to env dq ge bucket
    actions_list_env = f"{GDP_ENVIRONMENT_NAME}_action_list"
    results = context.run_validation_operator(
        actions_list_env, assets_to_validate=[dataset], run_id=ge_run_id
    )

    # build HTML pages out of the run and write to target env bucket
    context.build_data_docs(site_names=[f"{GDP_ENVIRONMENT_NAME}_docs"])

    # 100% success or bust
    suite_identifier = ExpectationSuiteIdentifier(suite_name)
    result_stats = results["details"][suite_identifier]["validation_result"][
        "statistics"
    ]
    LOGGER.info(
        "Source Validation Test against %s.%s Completion Stats: %s",
        schema,
        table,
        result_stats,
    )

    if not results["success"]:
        raise AirflowException(
            f"Source Validation Failed for: {schema}.{table} @ {result_stats['success_percent']}%"
        )
