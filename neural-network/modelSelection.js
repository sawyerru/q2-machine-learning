const tfvis = require('@tensorflow/tfjs-vis');
const NeuralNetwork = require('./neural-net');

/**
 * Automatically Select a model based on the data. Uses:
 *  Bias-Variance Tradeoff, 
 *  Testing/Validation/Test sets, 
 *  Hyperparameter tuning, 
 *  Regularization
 * 
 * to provide feedback on models and suggestions to improve
 */

 /**
  * Run test data through Neural Net to determine correct
  * @param {*} model 
  * @param {*} Xtest 
  * @param {*} Ytest 
  * @param {*} labelType 
  */
function runTest(model, Xtest, Ytest, labelType='categorical') {
    const Y = [];
    const X = [];
    const preds = [];
    let correctCount = 0;
    const n = Xtest.shape[0];
    for(let i = 0; i < n; i++) {

        const sample = Xtest.slice([i, 0], [1, Xtest.shape[1]]);
        const label = Ytest.slice([i, 0], [1, Ytest.shape[1]]).dataSync();

        X.push( sample.dataSync() );
        Y.push( label );
        
        const pred = model.predict(sample);

        if (labelType === 'categorical' && pred === label.indexOf(1)) correctCount++;
        else if ( labelType === label ) correctCount++;

        preds.push(pred);
    }

    return correctCount / n;
}

/**
 * Run Training and Validation Datasets to determine loss and accuracy metrics for the particular model
 * @param {*} buildConfig 
 * @param {*} Xtrain 
 * @param {*} Ytrain 
 * @param {*} Xvalid 
 * @param {*} Yvalid 
 * @param {*} Xtest 
 * @param {*} Ytest 
 */
async function runTrainAndValidation(buildConfig, Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest) {
    const trainingNet = new NeuralNetwork();
    trainingNet.build(buildConfig);
    await trainingNet.train(Xtrain, Ytrain);
    let training_loss = [];
    training_loss = trainingNet.trainingLogs.history.loss
        .map((y, x) => ({ x, y, }));;

    const validationNet = new NeuralNetwork();
    validationNet.build(buildConfig);
    let validation_loss = [];
    await validationNet.train(Xvalid, Yvalid);
    validation_loss = validationNet.trainingLogs.history.loss
        .map((y, x) => ({ x, y, }));;

    const acc = runTest(trainingNet, Xtest, Ytest, 'categorical');
    
    trainingNet.clean(); 
    validationNet.clean();

    const surface = { name: 'Line chart', tab: 'Charts' };
    const series = ['training_loss', 'validation_loss'];
    const plot = { values: [training_loss, validation_loss], series };
    tfvis.render.linechart(surface, plot);

    return correct / Xtest.shape[0];
}

module.exports = {
    runTrainAndValidation,
    runTest
}