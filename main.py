import sys
import models




if __name__ == "__main__":
    """
    #random forest
    model = models.RandomForest()
    output, accuracy = model[0], model[1]
    print(f"random forest: {accuracy:.2f}")


    #Gradient Boosting
    model = models.GradientBoosting()
    output, accuracy = model[0], model[1]
    print(f"Gradient Boosting Accuracy: {accuracy:.2f}")

    #SVM
    model = models.SupportVectorMachine()
    output, accuracy = model[0], model[1]
    print(f"SVM Accuracy: {accuracy:.2f}")

    #K nearest neighbors
    model = models.KNearestNeighbors()
    output, accuracy = model[0], model[1]
    print(f"K nearest neighbors Accuracy: {accuracy:.2f}")

    #Decision Tree
    model = models.DecisionTree()
    output, accuracy = model[0], model[1]
    print(f"Decision Tree Accuracy: {accuracy:.2f}")
    """

    #adaboost
    model = models.AdaBoost()
    output, accuracy = model[0], model[1]
    print(f"adaboost Accuracy: {accuracy:.2f}")

    output.to_csv('submission.csv', index=False)
