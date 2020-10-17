from great_expectations.core.id_dict import PartitionDefinition
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition


def test_partition():
    partition = Partition(
        name="test",
        data_asset_name="fake",
        definition=PartitionDefinition({"name": "hello"}),
        data_reference="nowhere"
    )
    # properties
    assert partition.name == "test"
    assert partition.data_asset_name == "fake"
    assert partition.definition == PartitionDefinition({"name": "hello"})
    assert partition.data_reference == "nowhere"

    assert str(test_partition) == str({
        "name": "test",
        "data_asset_name": "fake",
        "definition": {"name": "hello"},
        "data_reference": "nowhere"}
    )
    # test __eq__()
    test_partition1 = Partition(
        name="test",
        data_asset_name="fake",
        definition=PartitionDefinition({"name": "hello"}),
        data_reference="nowhere"
    )
    test_partition2 = Partition(
        name="test",
        data_asset_name="fake",
        definition=PartitionDefinition({"name": "hello"}),
        data_reference="nowhere"
    )
    test_partition3 = Partition(
        name="i_am_different",
        data_asset_name="fake",
        definition=PartitionDefinition({"name": "hello"}),
        data_reference="nowhere"
    )
    assert test_partition1 == test_partition2
    assert test_partition1 != test_partition3

    # test __hash__()
    assert test_partition.__hash__() == test_partition.__hash__()
    assert test_partition1.__hash__() == test_partition2.__hash__()
    assert test_partition1.__hash__() != test_partition3.__hash__()

