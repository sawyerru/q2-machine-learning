/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */

const BaseModel = require('../base-model');
const tf = require('@tensorflow/tfjs');

/** @private
 * Calculate the euclidian distance from 2 points based on all features of the points
 * @param {[]} point point 1 to measure
 * @param {[]} centroid point 2 to measure
 */
function euclideanDistance(point, centroid) {
  if (point.length !== centroid.length) {
    throw new Error('Dimension mismatch - cannot find distance of points');
  }
  let sum = 0;
  for (let i = 0; i < point.length; i++) {
    sum += Math.pow(point[i] - centroid[i], 2);
  }
  return Math.sqrt(sum);
}

/**
 * @private
 * Get the max and min of each feature
 * @param {Tensor} dataSet 
 * @returns array of features max and mins (d x 2)
 */
function getDataRange(dataSet) {
  const feats = []
  const [n, dimensions] = dataSet.shape;
  // get max and mins for each feature used to initialize clusters
  for (let d = 0; d < dimensions; d++ ) {
    const col = dataSet.slice([0,d], [n,1]);
    feats.push([tf.max(col).dataSync()[0], tf.min(col).dataSync()[0]])
  }
  return feats;
}

/**
 * @private
 * Initialize centroids using randomness between max and min
 * @param {Number} numCentroids 
 * @param {Array} featRange
 * @returns centroids array (k x d) 
 */
function intializeCentroids(numCentroids, featRange) {
  const centroids = [];
  const dims = featRange.length;
  for (let k = 0; k < numCentroids; k++ ) {
    const clust = []
    for (let d = 0; d < dims; d++) {
      const max = featRange[d][0];
      const min = featRange[d][1];
      clust.push(Math.random() * (max - min) + min);
    }
    centroids.push(clust);
  }
  return centroids;
}

/**
 * @private
 * Adjust the locations of the centroids
 * @param {Array} data 
 * @param {Array} clusters 
 * @returns [clusters, flag indicator]
 */
function adjustCentroidMean(data, clusters) {
  let isChanged = true;
  const dims = data[0].length;

  // split data into dict cluster: matrix
  const splitData = {}
  clusters.forEach( (ele, idx) => {
    splitData[idx] = [];
  });

  data.forEach( (ele, idx) => {
    const label = ele[dims - 1];
    splitData[label].push(ele); 
  });

  // Find mean for each dimension and for each cluster
  const means = {};
  for (let i = 0; i < clusters.length; i++) {
    means[i] = []
    for (let d = 0; d < dims - 1; d++) { // avoid label column
      let sum = 0;
      splitData[i].forEach( ele => {
        sum += ele[d]
      });
      means[i].push(sum/splitData[i].length);
    }
  }
  
  // update cluster values
  clusters.forEach( (ele, idx) => {
    for (let d = 0; d < dims - 1; d++) {
      if (ele[d] == means[idx][d]) { // calc mean and current mean are same 
        isChanged = false;
      }
      ele[d] = means[idx][d];
    }
  });

  return [clusters, isChanged]
}

/**
 * @private 
 * Assign all samples a centroid based on the minimum Euchlidean Distance
 * @param {*} data 
 * @param {*} clusters 
 * @returns data array
 */
function assignCentroidToPoints(data, clusters) {
  if (typeof(data[0]) === Number ) {
    throw new Error('Data is not 2D');
  }
  const n = data.length
  const dims = data[0].length;
  for (let i = 0; i < n; i++) {
    const distances = [];
    let sample = data[i];
    sample.pop();

    // get min distance compared to all centroids
    for (let k = 0; k < clusters.length; k++) {
      const dist = euclideanDistance(sample, clusters[k]);
      distances.push(dist);
    }
    const assignment = distances.indexOf(Math.min(...distances));
    
    // update dataset at sample at assignment column
    data[i][dims - 1] = assignment;
  }
  return data;
}


/** @public
 * K-Means Clustering algorithm to help group unknown data
 */
class ClusterClassifier extends BaseModel {
    constructor() {
      super();
      this._featureRanges = []
      this.centroids = []
      this.data = []
    }

    /**
     * @public
     * Train the Clustering to converge centroid locations
     * @param {Tensor} dataSet Feature Tensor only!
     * @param {Number} numClusters k number of clusters 
     */
    async train(dataSet, numClusters) {
      const [n, dimensions] = dataSet.shape;
      this._featureRanges = getDataRange(dataSet);
      this.centroids = intializeCentroids(numClusters, this._featureRanges);

      // Append -1 assignment column
      dataSet = tf.concat([dataSet, tf.fill([n, 1], -1)], 1);
      this.data = dataSet.arraySync();      
      // Loop over all samples and assign a label
      let isChanged = true;
      while (isChanged) {
        // Assign points to nearest Centroid
        this.data = assignCentroidToPoints(this.data, this.centroids);
        // Move Mean of clusters
        [this.centroids, isChanged] = adjustCentroidMean(this.data, this.centroids);
      }
    }

    /**
     * @public
     * predict the given sample based on the converged centroid locations
     * @param {Array} sample 
     * @return the prediction of which centroid is associated
     */
    predict(sample) {
      const distances = [];
      this.centroids.forEach( ele => {
        distances.push(euclideanDistance(sample, ele));
      });
      return distances.indexOf(Math.min(...distances));
    }
}

module.exports = ClusterClassifier;