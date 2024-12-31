from src.models.model_experiment import german_experiment
from src.models.model_repair_two_parts import calclulate_means_from_existing
from src.models.model_dnn1 import tf_run_experiments

if __name__ == "__main__":                                                                                  
    # german_experiment('knn', None)
    # german_experiment('decision-tree', None)
    # german_experiment('logistic-regression', None)
    # german_experiment('random-forest')
    # german_experiment('perceptron')
    tf_run_experiments('dnn2')
    tf_run_experiments('dnn3')
    
