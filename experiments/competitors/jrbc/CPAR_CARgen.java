/* -------------------------------------------------------------------------- */
/*                                                                            */
/* CPAR (CLASSIFICATION BASED ON PREDICTIVE ASSOCIATION RULES) CAR GENERATOR  *//*                                                                            */
/*                               Frans Coenen                                 */
/*                                                                            */
/*                         Wednesday 11 February 2004                         */
/*                                                                            */
/*                       Department of Computer Science                       */
/*                        The University of Liverpool                         */
/*                                                                            */
/* -------------------------------------------------------------------------- */

/* Class structure

AssocRuleMining
      |
      +-- Classification
      		|
		+-- PRM_CARgen
      			|
			+-- CPAR_CARgen		*/

// Java packages


/** Methods to produce classification rules using CPAR (Classification based on
Predictive Association Rules) algorithm first proposed by Xiaoxin Yin and 
Jiawei Han. Assumes that input dataset is organised such that classifiers are at 
the end of each record. Note: number of classifiers value is stored in the 
<TT>numClasses</TT> field.
@author Frans Coenen
@version 11 February 2004 */

/* To compile: javaARMpackc.exe CPAR_CARgen.java    */

import static java.lang.Math.toIntExact;

public class CPAR_CARgen extends PRM_CARgen {

	/* ------ FIELDS ------ */

	// Constants

	/** Gain similarity ratio */
	private double GAIN_SIMILARITY_RATIO = 0.99;

	/* ------ CONSTRUCTORS ------ */

	/** Constructor processes command line arguments.
    @param args the command line arguments (array of String instances). */

	public CPAR_CARgen(String[] args) {
		//super(args);
		// Process command line arguments
		for(int index=0; index<args.length; index=index+2)  {
			idArgument(args[index], args[index + 1]);
		}

		// If command line arguments read successfully (errorFlag set to "true")
		// check validity of arguments
		if (errorFlag) CheckInputArguments();
		else outputMenu();
	}

	protected void idArgument(String argument, String value) {
		if (argument.charAt(0) == '-') {
			char flag = argument.charAt(1);
			switch (flag) {
			case 'f':
				fileName = value;
				break;
			case 'n':
				numClasses = Integer.parseInt(value); 
				break;
			case 'w':
				TOTAL_WEIGHT_FACTOR = Double.parseDouble(value);
				break;
			case 'd':
				DECAY_FACROR = Double.parseDouble(value);
				break;
			case 's':
				GAIN_SIMILARITY_RATIO = Double.parseDouble(value);
				break;
			case 'g':
				MIN_BEST_GAIN = Double.parseDouble(value);
				break;
			default:
				System.out.println("INPUT ERROR: Unrecognised command line  argument -" + flag + argument);
				errorFlag = false;  
			}     
		} else {
			System.out.println("INPUT ERROR: All command line arguments must commence with a '-' character (" + argument + ")");
			errorFlag = false;
		}
	}

	/* ------ METHODS ------ */

	/* START CLASSIFICATION */
	
	public void fit() {
		// Set rule list to null. Note that startRuleList is defined in the
		// AssocRuleMining parent class and is also used to store Association
		// Rules (ARS) with respect to ARM.
		currentRlist.startRulelist = null;

		// Set DataArray and number of classes fields
		currentRlist.setDataArray(dataArray);
		currentRlist.setNumClasses(numClasses);

		// Check for classifier array
		if (classifiers==null) {
			System.out.println("ERROR: no classifiers array! To create a " +
					"classifiers array use createClassifiersArray(), " +
					"contained in ClassAprioriTserial class, called " +
					"from the application class.");
			System.exit(1);
		}

		// Start PRM generation process
		startCPAR();

		return;
	} 	

	/** Starts classification rule generation process.
    @return The classification accuarcy (%).		*/

	public double startClassification() {
		System.out.println("START CPAR CLASSIFICATION\n" +
				"------------------------");

		// Set rule list to null. Note that startRuleList is defined in the
		// AssocRuleMining parent class and is also used to store Association
		// Rules (ARS) with respect to ARM.
		currentRlist.startRulelist = null;

		// Set DataArray and number of classes fields
		currentRlist.setDataArray(dataArray);
		currentRlist.setNumClasses(numClasses);

		// Check for classifier array
		if (classifiers==null) {
			System.out.println("ERROR: no classifiers array! To create a " +
					"classifiers array use createClassifiersArray(), " +
					"contained in ClassAprioriTserial class, called " +
					"from the application class.");
			System.exit(1);
		}

		// Start CPAR generation process
		startCPAR();

		// Process rules
		processRules();

		// Test classification using the test set and return accuracy.
		return(twoDecPlaces(testClassification()));
	} 	

	/* START CPAR CLASSIFICATION */

	/** Commences CPAR process to generate CARs. <P> Proceeds as follows for
    each class:
    1) Generate attribute array (for computational convenience only).
    2) Generate P and N example data sets.
    2) Calculate weighting threshold.
    3) While current weighting for P is greater than threshold:
        a) Copy P, N and A to: get P', N' and A'.
        b) If first iteration copy PN to PN', else combine PN' positive weightings
           from previous iteration and original negative weightings to form
           new PN array.
        c) If not possible to generate further antecedent attributes break.
        d) Generate rule.		 */

	private void startCPAR() {
		// Generate attribute array
		attributes = new double[numOneItemSets-numClasses+1][2];
		for(int attIndex=0;attIndex<attributes.length;attIndex++) {
			attributes[attIndex][0]=0.0;
			attributes[attIndex][1]=0.0;
		}

		// Loop through classifiers array and find rule for each
		for (int index=0;index<classifiers.length;index++) {
			// Generate positive and negative examples from training set and
			// PN array
			generatePosAndNegExamples(classifiers[index]);
			generatePNarray();

			// calculate start total weight threshold of positive examples
			double totalWeightThreshold = TOTAL_WEIGHT_FACTOR*
					getTotalWeighting(positiveExamples);

			// Identify consequent of rule	
			long[] consequent = new long[1];
			consequent[0] = classifiers[index];

			// Process examples datasets until total weighting reduced to
			// below threshold.
			while (getTotalWeighting(positiveExamples)> totalWeightThreshold) {
				// Copy P, N, A and PN to P', N', A' and PN'
				positiveExamples2 = copyExamplesArray(positiveExamples);
				negativeExamples2 = copyExamplesArray(negativeExamples);
				pn_array2=copyDouble2Darray(pn_array);
				attributes2 = copyDouble2Darray(attributes);
				// It is possible, if no attributes with gain above the
				// specified minimum are discovered, that the current totalWeight
				// for P will never be reduced to below the threshold --- therefore
				// tesy for this situation.
				if (noValidGainsinPNarray()) break;
				// Proceed with generation process
				cparGeneration(null,consequent);
			}
		}
	}

	/* CPAR GENERATION */
	/** Uses the CPAR algorithm to generate a CAR for the given class by
    examining gain for each available attribute.
    @param antecedent the rule antecedent sofar.
    @param consequent the given class. */

	private void cparGeneration(long[] antecedent, long[] consequent) {
		// Determine weighting of previous example data sets.
		double oldPosWeight = (double) getWeightOfExamples(positiveExamples2);
		double oldNegWeight = (double) getWeightOfExamples(negativeExamples2);

		// Loop through attributes array and determine gain if added to rule,
		// gains placed in second row of attribute array.
		calculateGains(oldPosWeight,oldNegWeight);

		// Get index for attribute with best gain
		int indexOfBestAttribute = getBestGain();    // Inherited method
		double bestGain = attributes2[indexOfBestAttribute][1];	

		// if best gain less than user specified minimum stop.
		if (bestGain <= MIN_BEST_GAIN) {
			foundRule(antecedent,consequent);
			return;
		}

		// Proceed
		cparGeneration2(bestGain,antecedent,consequent);
	}

	/* CPAR GENERATION 2 */
	/** Continues to process example data sets using CPAR algorithm. <P> loop
    through attribute array finding gains above the gain threshold and then
    processes the associated attribute (there mat be several such attributes
    or none).
    @param bestGain the highest gain value found on this iteration.
    @param antecedent the rule antecedent sofar.
    @param consequent the given class. */

	private void cparGeneration2(double bestGain, long[] antecedent, long[] consequent) {
		// Determine gain threshold above which a gain value is considered
		// to be sufficiently high to be considered (provided that it is
		// also above the minimum user specified threshold).
		double gainThreshold = bestGain*GAIN_SIMILARITY_RATIO;
		if (gainThreshold < MIN_BEST_GAIN) gainThreshold=MIN_BEST_GAIN;	

		//Loop through attributes array
		for (int index=1;index<attributes2.length;index++) {
			// If: (i) attribute not already used in rule antecedent, and (ii)
			// gain is above gain threshold, include in rule antecedent.
			if (attributes2[index][0]==0.0 && attributes2[index][1]>gainThreshold) {
				ExamplesStruct[] tempPosExamples = copyExamplesArray(positiveExamples2);
				ExamplesStruct[] tempNegExamples = copyExamplesArray(negativeExamples2);
				double[][] temp_pn_array = copyDouble2Darray(pn_array2);
				double[][] temp_attributes = copyDouble2Darray(attributes2);
				
				cparGeneration3((long) index, antecedent, consequent);
				
//				ExamplesStruct[] positiveExamples2 = copyExamplesArray(tempPosExamples);
//				ExamplesStruct[] negativeExamples2 = copyExamplesArray(tempNegExamples);
//				double[][] pn_array2 = copyDouble2Darray(temp_pn_array);
//				double[][] attributes2 = copyDouble2Darray(temp_attributes);
				
				positiveExamples2 = copyExamplesArray(tempPosExamples);
				negativeExamples2 = copyExamplesArray(tempNegExamples);
				pn_array2 = copyDouble2Darray(temp_pn_array);
				attributes2 = copyDouble2Darray(temp_attributes);
			}
		}
	}

	/* CPAR GENERATION 3 */
	/** Process the given attribute using CPAR algorithm. <P> Adds attribute to
    rule so far and repeats.
    @param attribute the attribute to be added to the current rule antecedent.
    @param antecedent the rule antecedent sofar.
    @param consequent the given class. */

	private void cparGeneration3(long attribute, long[] antecedent,
			long[] consequent) {

		// Confirm by assigning to antecedent and indicating, in attributes
		// array, that selected attribute is no longer available
		antecedent = reallocInsert(antecedent,attribute);
		attributes2[Math.toIntExact(attribute)][0]=1.0;

		// Revise example datasets by removing examples that do not satisfy
		// the current rule antecedent
		positiveExamples2 = removeExDoNotSatRule(0,antecedent,positiveExamples2);
		negativeExamples2 = removeExDoNotSatRule(1,antecedent,negativeExamples2);

		// If the negative example rule set is empty the total weighting for
		// the set will be 0.0 and thus any gain calculated for any attribute
		// will also be 0.0 therefore in the event of an empty negative example
		// set we may as well stop here rather than recur and calculate a set
		// of gains which will not be above the threshold!
		if (negativeExamples2==null) {
			foundRule(antecedent,consequent);
			return;
		}

		// Repeat
		cparGeneration(antecedent,consequent);
	}

	/* ------------------------------------------------------------ */
	/*                                                              */
	/*                              OUTPUT                          */
	/*                                                              */
	/* ------------------------------------------------------------ */

	/* OUTPUT SETTINGS */

	/** Outputs command line values provided by user. (Overrides higher level
    method.) */

	protected void outputSettings() {
		System.out.println("SETTINGS\n----------");
		System.out.println("File name             = " + fileName);
		System.out.println("Num. classes          = " + numClasses);
		System.out.println("K value               = " + K_VALUE);
		System.out.println("Min. best gain        = " + MIN_BEST_GAIN);
		System.out.println("Total Weight Factor   = " + TOTAL_WEIGHT_FACTOR);
		System.out.println("Decay factor          = " + DECAY_FACROR);
		System.out.println("Gain similarity ratio = " + GAIN_SIMILARITY_RATIO);
		System.out.println();
	}
}

