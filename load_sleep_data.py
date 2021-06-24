from beetl.task_sleep import BeetlSleepDataset
ds = BeetlSleepDataset()
ds.download()

# Load all subject data
X, y, info = ds.get_data()

# Assume source group is subject 0-4, target group is subject 5-7,
# and subject 8,9 are from target group for testing.
X_source_train, y_source_train, info = ds.get_data(subjects=range(5))
X_target_train, y_target_train, info = ds.get_data(subjects=range(5, 8))
X_target_test, y_target_test, _ = ds.get_data(subjects=range(8, 10))
