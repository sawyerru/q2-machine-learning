/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */

const tf = require('@tensorflow/tfjs');
/**
 * @private
 * @abstract
 * Base model class for all Machine Learning algorithms
 * In Development, not entirely sure if I can bring more functions into it or break some into another base abstract class
 */
class BaseModel{
    constructor() {
        //< Private
        this._model = undefined;
        this._epochs = 100;
        this._batchSize = 32;
        this._verbose = 2;
        this._config;
    }

    /**
     * @public
     * Display model summary in console if a Tensorflow model, otherwise will just return the model defined by the child class
     * @returns the model
     */
    getModel() {
        if (this._model === undefined) {
            throw Error('Model undefined. please build your model first with build().')
        }
        if (this._model.name !== undefined) {
            this._model.summary();
        }
        return this._model;

    }

    /** 
     * @public
     * Use after any model runs to clear local memory and run efficiently.
     * Only applicable to Tensorflow powered models (Neural Net and Linear Regression), but won't cause errors
     */
    clean() {
        tf.disposeVariables();
    }   
}

module.exports = BaseModel;