# Q2 Machine Learning
This module is used for numerous unsupervised and supervised learning algorithms. Below is a list of included and working algorithms
* K-Means Clustering
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes Classifier
* Linear Regression 
* Neural Network 
* Principle Component Analysis
* Data Manipulator (for data transformations, encoding, normalization, etc.)

#### To Be Implemented:
- Apriori Rules Engine
- Support Vector Machine 

### How to Install
`npm install q2-machine-learning`

### How to Call
`import {<class>} from 'q2-machine-learning'`
Possible <class> options:
- DataManipulator
- ClusterClassifier
- DecisionTreeClassifier
- RandomForestClassifier
- LinearRegression
- NaiveBayesClassifier
- NeuralNetwork
- PCASelector
- AprioriAssociationGenerator
- SupportVectorMachine

For NeuralNetwork you can also import 2 supporting functions to inform model selection
- runTrainAndValidation
- runTest

## You can find examples of how each Class can function work in examples/
### Run tests
`npm test`

### Examples are found in tests/

