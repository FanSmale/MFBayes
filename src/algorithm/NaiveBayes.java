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
	 * Calculate the class distribution with Laplacian smooth.
	 ********************
	 */
	public void calculateClassDistribution() {
		classDistribution = new double[numClasses];
		classDistributionLaplacian = new double[numClasses];
		
		double[] tempCounts = new double[numClasses];
		for (int i = 0; i < numInstances; i++) {
			int tempClassValue = (int)data.instance(i).classValue();
			tempCounts[tempClassValue] ++;
		}//Of for i
		
		for (int i = 0; i < numClasses; i++) {
			classDistribution[i] = tempCounts[i] / numInstances;
			classDistributionLaplacian[i] = (tempCounts[i] + 1) / (numInstances + numClasses);
		}//Of for i
		
		System.out.println("Class distribution: " + Arrays.toString(classDistribution));
		System.out.println("Class distribution Laplacian: " + Arrays.toString(classDistributionLaplacian));
	}//Of calculateClassDistribution
	
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
					conditionalProbabilitiesLaplacian[i][j][k] = (conditionalProbabilities[i][j][k] + 1)
							/ (tempCount + numClasses);
					conditionalProbabilities[i][j][k] /= tempCount;
				} // Of for k
			} // Of for j
		} // Of for i

		System.out.println(Arrays.deepToString(conditionalProbabilities));
	}// Of calculateConditionalProbabilities

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
		// Find the biggest one
		double tempBiggest = -100;
		int resultBestIndex = 0;
		for (int i = 0; i < numClasses; i++) {
			double tempPseudoProbability = Math.log(classDistributionLaplacian[i]);
			for (int j = 0; j < numConditions; j++) {
				int tempAttributeValue = (int) paraInstance.value(j);

				// Laplacian smooth.
				tempPseudoProbability += Math.log(conditionalProbabilities[i][j][tempAttributeValue]);
			} // Of for j
			
			if (tempBiggest < tempPseudoProbability) {
				tempBiggest = tempPseudoProbability;
				resultBestIndex = i;
			}//Of if
		} // Of for i

		return resultBestIndex;
	}// Of classify

	/**
	 ********************
	 * Compute accuracy.
	 ********************
	 */
	public double computeAccuracy() {
		double tempCorrect = 0;
		for (int i = 0; i < numInstances; i++) {
			if (predicts[i] == (int)data.instance(i).classValue()){
				tempCorrect ++;
			}//Of if
		} // Of for i
		
		double resultAccuracy = tempCorrect / numInstances;
		return resultAccuracy;
	}// Of computeAccuracy
	
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
		System.out.println("Hello, Naive Bayes. I only want to test the constructor.");
		String tempFilename = "src/data/mushroom.arff";
		// String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		NaiveBayes tempLearner = new NaiveBayes(tempFilename);
		tempLearner.calculateClassDistribution();
		tempLearner.calculateConditionalProbabilities();
		tempLearner.classify();
		
		System.out.println("The accuracy is: " + tempLearner.computeAccuracy());
	}// Of main
}// Of class NaiveBayes