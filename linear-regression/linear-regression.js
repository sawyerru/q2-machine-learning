/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */
const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');

const BaseModel = require('../base-model');

/**
 * Linear Regression model implemented with Tensorflow Neural Net,
 * No activation, no hidden layers. Only input, bias, and output.
 */
class LinearRegression extends BaseModel {
    constructor() {
        super();
        //< Public
        this.trainingLogs; 
    }

    /**
     * @public
     * Getter to retrive the weights and bias from trained model
     * @returns {Array} [ Array of weights from input, bias term]
     */
    getWeightsAndBias() {
        if (this._model === undefined) {
            throw Error('Model not created yet, please use build() function first');
        } else {
            const weights = this._model.layers[0].getWeights()[0].dataSync();
            const bias = this._model.layers[0].getWeights()[1].dataSync();
            return [weights, bias]
        }
        
    }

    /**
     * @public
     * Build the model based on user defined parameters contained by the config object.
     * With no parameter the function creates a linear regression model
     * @param {Object} config 
     */
    build(inputDimensions) {
        // Build user defined neural net
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: 1, 
            useBias: true,
            inputShape: [inputDimensions],
        }));

        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });
        
        this._model = model;
    }

    /**
     * @public
     * Train the linear regression model using features and labels. Call is asynchronous please use 'await' keyword
     * ( ex. await linearRegression.train(X, Y); )
     * @param {Tensor} X 
     * @param {Tensor} Y 
     * @param {HTML Element | tf.visor} surface to display chart on
     */
    async train(X, Y, surface= undefined) {
        const trainLogs = [];
        const start = new Date().getTime();
        const trainingObject = await this._model.fit(X, Y,
            {
            epochs: this._epochs,
            verbose: this._verbose, 
            batchSize: this._batchSize, 
            // callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'}),
            callbacks: {
                onEpochEnd: (epoch, log) => {
                    if (surface !== undefined ) {
                        trainLogs.push(log);
                        tfvis.show.history(surface, trainLogs, ['loss', 'acc'])
                        console.log(`Epoch ${epoch} yeilded`, log);
                    }
                },
                onTrainBegin: (log) => {
                    console.log('Linear Regression Beginning Training')
                },
                onTrainEnd: (log) => {
                    const end = new Date().getTime();
                    const runtime = (end  - start) / 1000;
                    console.log(`Training Finished in ${runtime}s`)
                },
            },
            shuffle: true,
        });

        this.trainingLogs = trainingObject;
    }

    /**
     * @public
     * Predict method to determine model output
     * @param {Array} X feature matrix, sample to be predicted based on model
     * @return {Number} prediction as number
     */
    predict(X) {
        let tensor = X; 
        if (Array.isArray(X) && X[0] === Number) {
            tensor = tf.tensor(X, [1, X.length]);
        }
        const pred = this._model.predict(tensor);

        return pred.dataSync()[0];
    }

}

module.exports = LinearRegression;
