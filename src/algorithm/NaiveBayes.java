package algorithm;

import java.io.FileReader;
import java.util.Arrays;

import common.*;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The Naive Bayes algorithm..
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The Bayes project.
 * <p>
 * Progress: The very beginning.<br>
 * Written time: April 4, 2020. <br>
 * Last modify time: April 4, 2020.
 */

public class NaiveBayes {

	/**
	 * The data.
	 */
	Instances data;

	/**
	 * The number of classes. For binary classification it is 2.
	 */
	int numClasses;

	/**
	 * The number of instances.
	 */
	int numInstances;

	/**
	 * The number of conditional attributes.
	 */
	int numConditions;

	/**
	 * The prediction, including queried and predicted labels.
	 */
	int[] predicts;

	/**
	 * Class distribution.
	 */
	double[] classDistribution;

	/**
	 * Class distribution with Laplacian smooth.
	 */
	double[] classDistributionLaplacian;

	/**
	 * The conditional probabilities for all classes over all attributes on all
	 * values.
	 */
	double[][][] conditionalProbabilities;

	/**
	 * The conditional probabilities with Laplacian smooth.
	 */
	double[][][] conditionalProbabilitiesLaplacian;

	/**
	 * The Guassian parameters.
	 */
	GaussianParamters[][] gaussianParameters;

	/**
	 * Data type.
	 */
	int dataType;

	/**
	 * Nominal.
	 */
	public static final int NOMINAL = 0;

	/**
	 * Numerical.
	 */
	public static final int NUMERICAL = 1;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 ********************
	 */
	public NaiveBayes(String paraFilename) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		data.setClassIndex(data.numAttributes() - 1);
		numConditions = data.numAttributes() - 1;
		numInstances = data.numInstances();
		numClasses = data.attribute(numConditions).numValues();
	}// Of the constructor

	/**
	 ********************
	 * Set the data type.
	 ********************
	 */
	public void setDataType(int paraDataType) {
		dataType = paraDataType;
	}// Of setDataType

	/**
	 ********************
	 * Calculate the class distribution with Laplacian smooth.
	 ********************
	 */
	public void calculateClassDistribution() {
		classDistribution = new double[numClasses];
		classDistributionLaplacian = new double[numClasses];

		double[] tempCounts = new double[numClasses];
		for (int i = 0; i < numInstances; i++) {
			int tempClassValue = (int) data.instance(i).classValue();
			tempCounts[tempClassValue]++;
		} // Of for i

		for (int i = 0; i < numClasses; i++) {
			classDistribution[i] = tempCounts[i] / numInstances;
			classDistributionLaplacian[i] = (tempCounts[i] + 1) / (numInstances + numClasses);
		} // Of for i

		System.out.println("Class distribution: " + Arrays.toString(classDistribution));
		System.out.println(
				"Class distribution Laplacian: " + Arrays.toString(classDistributionLaplacian));
	}// Of calculateClassDistribution

	/**
	 ********************
	 * Calculate the conditional probabilities with Laplacian smooth.
	 ********************
	 */
	public void calculateConditionalProbabilities() {
		conditionalProbabilities = new double[numClasses][numConditions][];
		conditionalProbabilitiesLaplacian = new double[numClasses][numConditions][];
		for (int i = 0; i < numClasses; i++) {
			for (int j = 0; j < numConditions; j++) {
				int tempNumValues = data.attribute(j).numValues();
				conditionalProbabilities[i][j] = new double[tempNumValues];
				conditionalProbabilitiesLaplacian[i][j] = new double[tempNumValues];
				// Scan once to obtain the total numbers
				int tempCount = 1;
				for (int k = 0; k < data.numInstances(); k++) {
					if ((int) data.instance(k).classValue() != i) {
						continue;
					} // Of if

					tempCount++;
					// Count for the probability
					int tempValue = (int) data.instance(k).value(j);
					conditionalProbabilities[i][j][tempValue]++;
				} // Of for k

				// Now for the real probability
				for (int k = 0; k < tempNumValues; k++) {
					// Laplacian smooth here.
					conditionalProbabilitiesLaplacian[i][j][k] = (conditionalProbabilities[i][j][k]
							+ 1) / (tempCount + numClasses);
					conditionalProbabilities[i][j][k] /= tempCount;
				} // Of for k
			} // Of for j
		} // Of for i

		System.out.println(Arrays.deepToString(conditionalProbabilities));
	}// Of calculateConditionalProbabilities

	/**
	 ********************
	 * Calculate the conditional probabilities with Laplacian smooth.
	 ********************
	 */
	public void calculateGausssianParameters() {
		gaussianParameters = new GaussianParamters[numClasses][numConditions];

		double[] tempValuesArray = new double[numInstances];
		int tempNumValues = 0;
		double tempSum = 0;

		for (int i = 0; i < numClasses; i++) {
			for (int j = 0; j < numConditions; j++) {
				tempSum = 0;

				// Obtain values for this class.
				tempNumValues = 0;
				for (int k = 0; k < numInstances; k++) {
					if ((int) data.instance(k).classValue() != i) {
						continue;
					} // Of if

					tempValuesArray[tempNumValues] = data.instance(k).value(j);
					tempSum += tempValuesArray[tempNumValues];
					tempNumValues++;
				} // Of for k

				// Obtain parameters.
				double tempMu = tempSum / tempNumValues;

				double tempSigma = 0;
				for (int k = 0; k < tempNumValues; k++) {
					tempSigma += (tempValuesArray[k] - tempMu) * (tempValuesArray[k] - tempMu);
				} // Of for k
				tempSigma /= tempNumValues;
				tempSigma = Math.sqrt(tempSigma);

				gaussianParameters[i][j] = new GaussianParamters(tempMu, tempSigma);
			} // Of for j
		} // Of for i
		
		System.out.println(Arrays.deepToString(gaussianParameters));
	}// Of calculateGausssianParameters

	/**
	 ********************
	 * Classify all instances, the results are stored in predicts[].
	 ********************
	 */
	public void classify() {
		predicts = new int[numInstances];
		for (int i = 0; i < numInstances; i++) {
			predicts[i] = classify(data.instance(i));
		} // Of for i
	}// Of classify

	/**
	 ********************
	 * Classify an instances.
	 ********************
	 */
	public int classify(Instance paraInstance) {
		if (dataType == NOMINAL) {
			return classifyNominal(paraInstance);
		} else if (dataType == NUMERICAL) {
			return classifyNumerical(paraInstance);
		} // Of if

		return -1;
	}// Of classify

	/**
	 ********************
	 * Classify an instances with nominal data.
	 ********************
	 */
	public int classifyNominal(Instance paraInstance) {
		// Find the biggest one
		double tempBiggest = -10000;
		int resultBestIndex = 0;
		for (int i = 0; i < numClasses; i++) {
			double tempPseudoProbability = Math.log(classDistributionLaplacian[i]);
			for (int j = 0; j < numConditions; j++) {
				int tempAttributeValue = (int) paraInstance.value(j);

				// Laplacian smooth.
				tempPseudoProbability += Math
						.log(conditionalProbabilities[i][j][tempAttributeValue]);
			} // Of for j

			if (tempBiggest < tempPseudoProbability) {
				tempBiggest = tempPseudoProbability;
				resultBestIndex = i;
			} // Of if
		} // Of for i

		return resultBestIndex;
	}// Of classifyNominal

	/**
	 ********************
	 * Classify an instances with numerical data.
	 ********************
	 */
	public int classifyNumerical(Instance paraInstance) {
		// Find the biggest one
		double tempBiggest = -10000;
		int resultBestIndex = 0;
		double tempSqrt2Pi = Math.log(2 * Math.PI) / 2;
		for (int i = 0; i < numClasses; i++) {
			double tempPseudoProbability = Math.log(classDistributionLaplacian[i]);
			for (int j = 0; j < numConditions; j++) {
				double tempAttributeValue = paraInstance.value(j);
				double tempSigma = gaussianParameters[i][j].sigma;
				double tempMu = gaussianParameters[i][j].mu;

				tempPseudoProbability += -tempSqrt2Pi - Math.log(tempSigma)
						- (tempAttributeValue - tempMu) * (tempAttributeValue - tempMu)
								/ (2 * tempSigma * tempSigma);
			} // Of for j

			if (tempBiggest < tempPseudoProbability) {
				tempBiggest = tempPseudoProbability;
				resultBestIndex = i;
			} // Of if
		} // Of for i

		return resultBestIndex;
	}// Of classifyNumerical

	/**
	 ********************
	 * Compute accuracy.
	 ********************
	 */
	public double computeAccuracy() {
		double tempCorrect = 0;
		for (int i = 0; i < numInstances; i++) {
			if (predicts[i] == (int) data.instance(i).classValue()) {
				tempCorrect++;
			} // Of if
		} // Of for i

		double resultAccuracy = tempCorrect / numInstances;
		return resultAccuracy;
	}// Of computeAccuracy

	/**
	 ************************* 
	 * Test nominal data.
	 ************************* 
	 */
	public static void testNominal() {
		System.out.println("Hello, Naive Bayes. I only want to test the nominal data.");
		String tempFilename = "src/data/mushroom.arff";
		// String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";

		NaiveBayes tempLearner = new NaiveBayes(tempFilename);
		tempLearner.setDataType(NOMINAL);
		tempLearner.calculateClassDistribution();
		tempLearner.calculateConditionalProbabilities();
		tempLearner.classify();

		System.out.println("The accuracy is: " + tempLearner.computeAccuracy());
	}// Of testNominal

	/**
	 ************************* 
	 * Test numerical data.
	 ************************* 
	 */
	public static void testNumerical() {
		System.out.println(
				"Hello, Naive Bayes. I only want to test the numerical data with Gaussian assumption.");
		//String tempFilename = "src/data/iris.arff";
		//String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";
		String tempFilename = "src/data/wdbc_norm_ex.arff";
		
		NaiveBayes tempLearner = new NaiveBayes(tempFilename);
		tempLearner.setDataType(NUMERICAL);
		tempLearner.calculateClassDistribution();
		tempLearner.calculateGausssianParameters();
		tempLearner.classify();

		System.out.println("The accuracy is: " + tempLearner.computeAccuracy());
	}// Of testNominal

	/**
	 ************************* 
	 * Test this class.
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		//testNominal();
		 testNumerical();
	}// Of main

	/**
	 ************************* 
	 * An inner class to store parameters.
	 ************************* 
	 */
	private class GaussianParamters {
		double mu;
		double sigma;

		public GaussianParamters(double paraMu, double paraSigma) {
			mu = paraMu;
			sigma = paraSigma;
		}// Of the constructor
		
		public String toString(){
			return "(" + mu + ", " + sigma + ")";
		}//Of toString
	}// Of GaussianParamters
}// Of class NaiveBayes