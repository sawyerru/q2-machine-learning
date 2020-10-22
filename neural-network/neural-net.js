/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */
const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');
const BaseModel = require('../base-model');


/**
 * Basic Neural Net - 
 * can be adaptable to any config object passed to build function
 */
class NeuralNetwork extends BaseModel {
    constructor() {
        super();

        //< Public
        this.trainingLogs; 
    }

    /**
     * @public
     * Build the model based on user defined parameters contained by the config object.
     * With no parameter the function creates a simple neural net. 
     * @param {Object} config contains {inputDims: number, architecture: array, outputClasses: number, activation function: string, optimizer function: string, loss function: string }
     */
    build(config) {
        // destructuring config object
        let {
            inputDims, 
            architecture = [], 
            outputClasses = 1,
            activationF = 'relu',
            optimizerF = tf.train.adam(),
            lossF = 'meanSquaredError',
            regularizer = tf.regularizers.l1l2()
        } = config;
        this._config = config;

        // Build user defined neural net
        const model = tf.sequential();
        for (let i = 0; i < architecture.length; i++) {
            if (i === 0) { // first hidden layer needs to define input shape
                model.add(tf.layers.dense({
                    units: architecture[i],
                    inputShape: [inputDims],
                    activation: activationF,
                }));
            } else if (regularizer !== null){
                model.add(tf.layers.dense({
                    units: architecture[i],
                    activation: activationF,
                    kernelRegularizer: regularizer
                }));
            } else {
                model.add(tf.layers.dense({
                    units: architecture[i],
                    activation: activationF,
                }));
            }
        }

        // Change activation
        if (outputClasses > 1 && lossF !== 'categoricalCrossentropy') {
            console.warn(`Converting final layer to accomodate outputClasses param = ${outputClasses}`)
            activationF = 'softmax';
            lossF = 'categoricalCrossentropy';
        }

        // Adding final output layer based on config
        model.add(tf.layers.dense({
            units: outputClasses,
            activation: activationF,
        }));

        // Compile with selected optimizer and loss function
        model.compile({
            optimizer: optimizerF,
            loss: lossF,
            metrics: ['accuracy']
        });
        
        this._model = model;
    }

    /**
     * @public
     * Train the neural net model, model must be built and this call is made asynchronously - use await keyword in call
     * ( ex. await neuralNet.train(X, Y); )
     * @param {Tensor} X Tensor Object of features
     * @param {Tensor} Y Tensor Object of labels 
     */
    async train(X, Y, surface=undefined) {
        if (this._model === undefined) {
            throw Error('Model not defined, use build() function before training')
        }
        const trainLogs = [];
        const start = new Date().getTime();
        const trainingObject = await this._model.fit(X, Y,
            {
            epochs: this._epochs,
            verbose: this._verbose, 
            batchSize: this._batchSize, 
            callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'}),
            callbacks: {
                onEpochEnd: (epoch, log) => {
                    if (surface !== undefined) {
                        trainLogs.push(log);
                        tfvis.show.history(surface, trainLogs, ['loss', 'acc'])
                        console.log(`Epoch ${epoch} yeilded`, log);
                    }
                },
                onTrainBegin: (log) => {
                    console.log('Neural Net Beginning Training')

                },
                onTrainEnd: (log) => {
                    const end = new Date().getTime();
                    const runtime = (end  - start) / 1000;
                    console.log(`Training Finished in ${runtime}s`);
                },
            },
            shuffle: true,
        });

        this.trainingLogs = trainingObject;
    }

    /**
     * @public
     * Predict method to determine model output, remembers from config on format of output
     * @param {Array} X feature vector, sample to be predicted based on model
     * @returns {Number} prediction will come back as a number for regression or a number to represent class
     */
    predict(X) {
        let tensor = X; 
        if (Array.isArray(X) && X[0] === Number) {
            tensor = tf.tensor(X, [1, X.length]);
        }
        const pred = this._model.predict(tensor);

        if (this._config.outputClasses > 1) {
            return pred.argMax(-1).dataSync()[0];
        } else {
            return pred.dataSync()[0];
        }
    }
}

module.exports = NeuralNetwork;