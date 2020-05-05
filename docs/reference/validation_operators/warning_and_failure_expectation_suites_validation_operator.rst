.. _warning_and_failure_expectation_suites_validation_operator:

================================================================================
WarningAndFailureExpectationSuitesValidationOperator
================================================================================

WarningAndFailureExpectationSuitesValidationOperator implements a business logic pattern that many data practitioners consider useful - grouping the expectations about a data asset into two expectation suites.

The "failure" expectation suite contains expectations that are considered important enough to stop the pipeline when they are violated. The rest of the expectations go into the "warning" expectation suite.


WarningAndFailureExpectationSuitesValidationOperator retrieves the two expectation suites ("failure" and "warning") for every data asset in the `assets_to_validate` argument of its `run` method. It does not require both suites to be present.

The operator invokes a list of actions on every validation result. The list is configured for the operator.
Each action in the list must be an instance of ValidationAction
class (or its descendants). Read more about actions here: :ref:`actions`.

After completing all the validations, it sends a Slack notification with the success status.


Configuration
--------------

Below is an example of this operator's configuration:

.. code-block:: yaml

    run_warning_and_failure_expectation_suites:
        class_name: WarningAndFailureExpectationSuitesValidationOperator

        # the following two properties are optional - by default the operator looks for
        # expectation suites named "failure" and "warning".
        # You can use these two properties to override these names.
        # e.g., with expectation_suite_name_prefix=boo_ and
        # expectation_suite_name_suffixes = ["red", "green"], the operator
        # will look for expectation suites named "boo_red" and "boo_green"
        expectation_suite_name_prefix="",
        expectation_suite_name_suffixes=["failure", "warning"],

        # optional - if true, the operator will stop and exit after first failed validation. false by default.
        stop_on_first_error=False,

        # put the actual webhook URL in the uncommitted/config_variables.yml file
        slack_webhook: ${validation_notification_slack_webhook}
        # optional - if "all" - notify always, "success" - notify only on success, "failure" - notify only on failure
        notify_on="all"

        # the operator will call the following actions on each validation result
        # you can remove or add actions to this list. See the details in the actions
        # reference
        action_list:
          - name: store_validation_result
            action:
              class_name: StoreValidationResultAction
              target_store_name: validations_store
          - name: store_evaluation_params
            action:
              class_name: StoreEvaluationParametersAction
              target_store_name: evaluation_parameter_store


Invocation
-----------

This is an example of invoking an instance of a Validation Operator from Python:

.. code-block:: python

    results = context.run_validation_operator(
        assets_to_validate=[batch0, batch1, ...],
        run_id="some_string_that_uniquely_identifies_this_run",
        validation_operator_name="operator_instance_name",
    )

* `assets_to_validate` - an iterable that specifies the data assets that the operator will validate. The members of the list can be either batches or triples that will allow the operator to fetch the batch: (data_asset_name, expectation_suite_name, batch_kwargs) using this method: :py:meth:`~great_expectations.data_context.BaseDataContext.get_batch`
* run_id - pipeline run id, a timestamp or any other string that is meaningful to you and will help you refer to the result of this operation later
* validation_operator_name you can instances of a class that implements a Validation Operator

The `run` method returns a result object.

The value of "success" is True if no critical expectation suites ("failure") failed to validate (non-critical warning") expectation suites are allowed to fail without affecting the success status of the run.

.. code-block:: json

    {
        "data_asset_identifiers": list of data asset identifiers
        "success": True/False,
        "failure": {
            expectation suite identifier: {
                "validation_result": validation result,
                "action_results": {action name: action result object}
            }
        }
        "warning": {
            expectation suite identifier: {
                "validation_result": validation result,
                "action_results": {action name: action result object}
            }
        }
    }


