
dataset_list = ['adult', 'compas', 'german', 'churn', 'compasm', 'whitewine', 'redwine']
black_box_list = ['RF', 'SVM', 'NN', 'DT', 'DNN']
neigh_list = ['random', 'genetic', 'rndgen', 'geneticp', 'rndgenp']
extr_list = ['DT', 'CPAR', 'FOIL', 'ANCHOR']
partition_list = ['2e', 'ts']

random_state = 42  # 0
test_size = 0.30

path = '/media/riccardo/data1/LOREM/'  #'/home/riccardo/Documenti/PhD/LOREM/'
path_data = path + 'dataset/'
path_partitions = path + 'partitions/'
path_models = path + 'blackbox_models/'
path_results = path + 'results/classification/'
path_neighbors = path + 'neighborhoods/'
