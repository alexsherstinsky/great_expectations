import click
import os

from .supporting_methods import (
    cli_message
)

msg_prompt_choose_data_source = """
Configure a DataSource?
    1. Directory on your local filesystem
    2. Relational database (SQL)
    0. Skip this step for now
"""

#     msg_prompt_choose_data_source = """
# Time to create expectations for your data. This is done in Jupyter Notebook/Jupyter Lab.
#
# Before we point you to the right notebook, what data does your project work with?
#     1. Directory on local filesystem
#     2. Relational database (SQL)
#     3. DBT (data build tool) models
#     4. None of the above
#     """


#     msg_prompt_dbt_choose_profile = """
# Please specify the name of the dbt profile (from your ~/.dbt/profiles.yml file Great Expectations \
# should use to connect to the database
#     """

#     msg_dbt_go_to_notebook = """
# To create expectations for your dbt models start Jupyter and open notebook
# great_expectations/notebooks/using_great_expectations_with_dbt.ipynb -
# it will walk you through next steps.
#     """

msg_prompt_filesys_enter_base_path = """
Please enter the path to the target directory.
The path may be either absolute (/Users/charles/my_data)
or relative to the project root directory (my_project_data)
"""

msg_prompt_datasource_name = """
Give your new data source a short name    
"""

msg_sqlalchemy_config_connection = """
Great Expectations relies on sqlalchemy to connect to relational databases.
Please make sure that you have it installed.         

Next, we will configure database credentials and store them in the "{0:s}" section
of this config file: great_expectations/uncommitted/credentials/profiles.yml:
"""

msg_unknown_data_source = """
We are looking for more types of data types to support. 
Please create a GitHub issue here: 
https://github.com/great-expectations/great_expectations/issues/new
In the meantime you can see what Great Expectations can do on CSV files.
To create expectations for your CSV files start Jupyter and open notebook
great_expectations/notebooks/using_great_expectations_with_pandas.ipynb - 
it will walk you through configuring the database connection and next steps. 
"""


def connect_to_datasource(context):
    data_source_selection = click.prompt(msg_prompt_choose_data_source, type=click.Choice(["1", "2", "0"]),
                                         show_choices=False)

    # if data_source_selection == "5": # dbt
    #     dbt_profile = click.prompt(msg_prompt_dbt_choose_profile)
    #     log_message(msg_dbt_go_to_notebook, color="blue")
    #     context.add_datasource("dbt", "dbt", profile=dbt_profile)
    if data_source_selection == "3":  # Spark
        path = click.prompt(
            msg_prompt_filesys_enter_base_path,
            default='/data/',
            type=click.Path(
                exists=True,
                file_okay=False,
                dir_okay=True,
                readable=True
            ),
            show_default=True
        )
        if path.startswith("./"):
            path = path[2:]

        default_data_source_name = os.path.basename(path)
        data_source_name = click.prompt(
            msg_prompt_datasource_name, default=default_data_source_name, show_default=True)

        # FIXME: The logic for this notebook needs to go to the top-level CLI
        # cli_message(msg_spark_go_to_notebook)
        context.add_datasource(data_source_name, "spark", base_directory=path)

    elif data_source_selection == "2":  # sqlalchemy
        data_source_name = click.prompt(
            msg_prompt_datasource_name, default="mydb", show_default=True)

        cli_message(
            msg_sqlalchemy_config_connection.format(data_source_name)
        )

        drivername = click.prompt("What is the driver for the sqlalchemy connection?", default="postgres",
                                  show_default=True)
        host = click.prompt("What is the host for the sqlalchemy connection?", default="localhost",
                            show_default=True)
        port = click.prompt("What is the port for the sqlalchemy connection?", default="5432",
                            show_default=True)
        username = click.prompt("What is the username for the sqlalchemy connection?", default="postgres",
                                show_default=True)
        password = click.prompt("What is the password for the sqlalchemy connection?", default="",
                                show_default=False, hide_input=True)
        database = click.prompt("What is the database name for the sqlalchemy connection?", default="postgres",
                                show_default=True)

        credentials = {
            "drivername": drivername,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database
        }
        context.add_profile_credentials(data_source_name, **credentials)

        # FIXME: The logic for this notebook needs to go to the top-level CLI
        # cli_message(msg_sqlalchemy_go_to_notebook, color="blue")

        context.add_datasource(
            data_source_name, "sqlalchemy", profile=data_source_name)

    elif data_source_selection == "1":  # csv
        path = click.prompt(
            msg_prompt_filesys_enter_base_path,
            type=click.Path(
                exists=False,
                file_okay=False,
                dir_okay=True,
                readable=True
            ),
            show_default=True
        )
        if path.startswith("./"):
            path = path[2:]

        default_data_source_name = os.path.basename(path)+"__local_dir"
        data_source_name = click.prompt(
            msg_prompt_datasource_name, default=default_data_source_name, show_default=True)

        context.add_datasource(data_source_name, "pandas", base_directory=path)

    else:
        cli_message(msg_unknown_data_source)  # , color="blue")
