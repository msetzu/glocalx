/* -------------------------------------------------------------------------- */
/*                                                                            */
/*                 PRM (PREDICTIVE RULE MINING) CAR GENERATOR                 *//*                                                                            */
/*                               Frans Coenen                                 */
/*                                                                            */
/*                          Tuesday 3 February 2004                           */
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
		+-- PRM_CARgen		*/

// Java packages


/** Methods to produce classification rules using PRM (Predictive Rule Mining)
algorithm first proposed by Xiaoxin Yin and Jiawei Han. Assumes
that input dataset is organised such that classifiers are at the end of each
record. Note: number of classifiers value is stored in the <TT>numClasses</TT>
field.
@author Frans Coenen
@version 3 February 2004 */

/* To compile: javaARMpackc.exe PRM_CARgen.java    */
	
import static java.lang.Math.toIntExact;

public class PRM_CARgen extends Classification {

	/* ------ FIELDS ------ */

	// Nested classes
	/** Structure to store an example (positive or negative) records with
    weighting. */
	protected class ExamplesStruct {
		/** Item set (example record). */
		long[] itemSet;
		/** Weighting for the itemset. */
		double weighting=1.0;

		/** One argument constructor
    @param record the itemset representing a record. */	
		protected ExamplesStruct(long[] record) {
			itemSet = copyItemSet(record);
		}

		/** Two argument constructor
    @param record the itemset representing a record.
    @param weight the weighting for the example record. */	
		protected ExamplesStruct(long[] record, double weight) {
			itemSet   = copyItemSet(record);
			weighting = weight;
		}
	}

	// Data structures
	/** 2-D array to hold positive examples */
	protected ExamplesStruct[] positiveExamples = null;
	/** 2-D array to hold negative examples */
	protected ExamplesStruct[] negativeExamples = null;
	/** 2-D array to temporaily hold positive examples, what Xiaoxin Yin and
    Jiawei Han refer to as P'. */
	protected ExamplesStruct[] positiveExamples2 = null;
	/** 2-D array to temporaily hold negative examples, what Xiaoxin Yin and
    Jiawei Han refer to as N'. */
	protected ExamplesStruct[] negativeExamples2 = null;
	/** 2-D PNarray for storing size of current negative and positive example
    sets, what Xiaoxin Yin and Jiawei Han refer to as A. */
	protected double[][] pn_array = null;
	/** 2-D  temporary PN array for storing size of current negative and
    positive example sets, what Xiaoxin Yin and Jiawei Han refer to as A'. */
	protected double[][] pn_array2 = null;

	// Constants
	/** Minimum total weight threshold */
//	protected final double TOTAL_WEIGHT_FACTOR=0.05;
	/** Weighting decrement */
//	protected final double DECAY_FACROR=1.0/3.0;
	
	protected double TOTAL_WEIGHT_FACTOR = 0.05;
	protected double DECAY_FACROR = 1.0/3.0;
	protected double MIN_BEST_GAIN = 0.7;

	/* ------ CONSTRUCTORS ------ */

	/** Constructor processes command line arguments.
    @param args the command line arguments (array of String instances). */
	public PRM_CARgen() {
		;
	}
			
			
	public PRM_CARgen(String[] args) {
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
//			argument = argument.substring(3,argument.length());
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
		startPRM();

		return;
	} 	

	/** Starts classification rule generation process.
    @return The classification accuracy (%).		*/

	public double startClassification() {
		System.out.println("START PRM CLASSIFICATION\n" +
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

		// Start PRM generation process
		startPRM();

		// Process rules
		processRules();

		// Test classification using the test set and return accuracy.
		return(twoDecPlaces(testClassification()));
	} 	

	/* START PRM CLASSIFICATION */

	/** Commences PRM process to generate CARs.  <P> Proceeds as follows for
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

	private void startPRM() {
		// Generate attribute array
		attributes = new double[numOneItemSets-numClasses+1][2];
		for(int attIndex=0;attIndex<attributes.length;attIndex++) {
			attributes[attIndex][0]=0.0;
			attributes[attIndex][1]=0.0;
		}

		// Loop through classifiers array and find rule for each
		for (int index=0;index<classifiers.length;index++) {
			// Generate positive and negative examples from training set and
			// a PN array
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
			while (getTotalWeighting(positiveExamples)>totalWeightThreshold) {
				// Copy P, N, PN array and attributes array
				positiveExamples2 = copyExamplesArray(positiveExamples);
				negativeExamples2 = copyExamplesArray(negativeExamples);
				pn_array2         = copyDouble2Darray(pn_array);
				attributes2       = copyDouble2Darray(attributes);
				// It is possible, if no attributes with gain above the
				// specified minimum are discovered, that the example positive
				// data set P will never be reduced to null --- therefore
				// tesy for this situation.
				if (noValidGainsinPNarray()) break;
				// Proceed with generation process
				prmGeneration(null,consequent);
			}
		}
	}

	/* PRM GENERATION */
	/** Uses the PRM algorithm to generate a CAR for the given class by
    examining gain for each available attribute.
    @param antecedent the rule antecedent sofar.
    @param consequent the given class.  */

	private void prmGeneration(long[] antecedent, long[] consequent) {
		// Determine total weighting of previous example data sets.
		double oldPosWeigth = (double) getWeightOfExamples(positiveExamples2);
		double oldNegWeigth = (double) getWeightOfExamples(negativeExamples2);

		// Loop through attributes array and determine gain if added to rule,
		// gains placed in second row of attribute array.
		calculateGains(oldPosWeigth,oldNegWeigth);


		// Get index for attribute with best gain
		int indexOfBestAttribute = getBestGain(); 	// Inheritted method
		double bestGain = attributes2[indexOfBestAttribute][1];	

		// If best gain less than minimum, stop.
		if (bestGain <= MIN_BEST_GAIN) {
			foundRule(antecedent,consequent);
			return;
		}

		// Confirm by assigning to antecedent and indicating, in attributes
		// array, that selected attribute is no longer available
		antecedent = reallocInsert(antecedent,(long) indexOfBestAttribute);
		attributes2[indexOfBestAttribute][0]=1.0;

		// Revise example datasets by removing examples that do not satisfy
		// the current rule antecedent
		positiveExamples2=removeExDoNotSatRule(0,antecedent,positiveExamples2);
		negativeExamples2=removeExDoNotSatRule(1,antecedent,negativeExamples2);

		// If the negative example rule set is empty the total weighting for
		// the set will be 0.0 and thus any gain calculated for any attribute
		// will also be 0.0, therefore in the event of an empty negative example
		// set we may as well stop here rather than recur and calculate a set
		// of gains which will not be above the threshold!
		if (negativeExamples2==null) {	
			foundRule(antecedent,consequent);
			return;
		}

		// Repeat
		prmGeneration(antecedent,consequent);
	}

	/* FOUND CLASSIFICATION RULE */	
	/** Inserts discovered rule into list of rules and revise positive examples
    array (not the copy of this array) and the PN array (again not the copy of
    this array). <P> Note the generic rule array also has field for support 
    and confidence that are not used here.
    @param antecedent the antecedent of the discovered rule.
    @param consequent the consequent of the discovered rule. */

	protected void foundRule(long[] antecedent, long[] consequent) {
		// It is possible, if we have no negative examples (N={}), to
		// produce a rule with an empty antecedent --- such a rule should
		// not be included in the rule list.
		if (antecedent==null) return;

		// Insert discovered rule into rule list
		double laplace = currentRlist.getLaplaceAccuracy(antecedent,
				consequent[0]);
		currentRlist.insertRuleintoRulelist(antecedent,consequent,laplace);

		// Revise positive examples
		revisePosExDoSatRule(antecedent,positiveExamples);
	}

	/* ------------------------------------------------------------ */
	/*                                                              */
	/*                       GAIN CALCULATIONS                      */
	/*                                                              */
	/* ------------------------------------------------------------ */

	/* CALCULATE GAINS */

	/** Loops through available attributes and determine gain for each if
    attribute added to rule. <P> Attribute is unavailable (i.e. has already been
    used in rule) if first element for attribute set to '1.0'
    @param oldPosWeigth total weighting represented by previous positive 
    examples.
    @param oldNegWeigth total weighting represented by previous negative 
    examples. */

	protected void calculateGains(double oldPosWeigth, double oldNegWeigth) {
		// Loop through attribute list
		for (int index=1;index<attributes2.length;index++) {
			// If attribute not selected previously determine gain.
			if ((int) attributes2[index][0] == 0) {
				// Get counts from PN-array
				double newPosWeigth=pn_array2[index][0];
				double newNegWeigth=pn_array2[index][1];
				// Calculate gain and add to attribute array
				attributes2[index][1] = calculateGain(newPosWeigth,
						newNegWeigth,oldPosWeigth,oldNegWeigth);
			}
		}
	}

	/* CALCULATE GAIN */

	/** Calculates gain for the given rule antecedent.
    @param posWeigth weighting represented by current positive examples.
    @param negWeight weighting represented by current negative examples.
    @param oldPosWeigth total weighting represented by previous positive 
    examples.
    @param oldNegWeigth total weighting represented by previous negative 
    examples.  */

	protected double calculateGain(double posWeigth, double negWeigth,
			double oldPosWeigth, double oldNegWeigth) {
		// If pos count is 0 gain is 0
		if ((int) posWeigth == 0) return(0.0);

		// Calculate gain
		double oldGain = Math.log(oldPosWeigth/(oldPosWeigth+oldNegWeigth));
		double newGain = Math.log((double) posWeigth/(double) (posWeigth+negWeigth));
		return(posWeigth*(newGain-oldGain));
	}

	/* ------------------------------------------------------------ */
	/*                                                              */
	/*                POSITIVE AND NEGATIVE EXAMPLES                */
	/*                                                              */
	/* ------------------------------------------------------------ */

	/* GENERATE POSITIVE AND NEGATIVE EXAMPLES */

	/** Generates positive (P) and negative (N) example data sets for the given
    class. <P> P = all records which DO include the current required class, (N) all
    records which DO NOT include the current class. D (Data Set) = P union N.
    @param classification the given class of interest. */

	protected void generatePosAndNegExamples(long classification) {
		int posIndex=0, negIndex=0;

		// Loop through data array (training set) and determine total number
		// of positive and negative records
		for(int index=0;index<dataArray.length;index++) {
			// get last index
			int lastIndex = dataArray[index].length-1;
			if (dataArray[index][lastIndex]==classification) posIndex++;
			else negIndex++;
		}

		// Dimension negative and positive arrays
		positiveExamples = new ExamplesStruct[posIndex];
		negativeExamples = new ExamplesStruct[negIndex];

		// Loop through data array again and add recordes to appropriate 
		// example set (positive or negative) 
		posIndex=0;
		negIndex=0;
		for(int index=0;index<dataArray.length;index++) {
			// get last index
			int lastIndex = dataArray[index].length-1;
			if (dataArray[index][lastIndex]==classification) {
				positiveExamples[posIndex] = new ExamplesStruct(dataArray[index]);
				posIndex++;
			}
			else {
				negativeExamples[negIndex] = new ExamplesStruct(dataArray[index]);
				negIndex++;
			}
		}
	}

	/* GET SIZE OF EXAMPLES DATA SET */
	/** Returns the size (length) of the given examples set.
    @param examples the given examples data set. */

	protected int getWeightOfExamples(ExamplesStruct[] examples) {
		if (examples==null) return(0);
		else return(examples.length);
	}

	/* REMOVE EXAMPLES THAT DO NOT SATISFY RULE */

	/** Remove from given example data sets all examples that DO NOT satisfy
    the given rule and adjust PN array copy accordingly. <P> Called whenever a 
    new attribute is added to the current rules antecedent. Creates a 
    new example structure array to hold records that DO staisfy the new rule
    rather than edditing the existing example structure array. Proceeds as 
    follows:
    <OL>
    <LI>Determine size of new example structure array.
    <LI>Dimension new example structure array.
    <LI>Populate new example structure array with records that DO sataisfy the 
    rule, and adjust PM array accordingly.
    </OL> 
    @param flag 0=positive examples, 1=negative examples.
    @param antecedent the given antecedent for the rule.
    @param examples the given examples data set.
    @return revise example data set with records that do not satisfy rule
    		removed. 		*/	

	protected ExamplesStruct[] removeExDoNotSatRule(int flag,
			long[] antecedent, ExamplesStruct[] examples) {
		// Return null if input array is empty.
		if (examples==null) return(null);

		// Dimension new array
		int size = getNumExDoSatRule(antecedent,examples);
		// If new array is to be empty all weightings in the copy of the PN
		// array associated with this example structure array must be 0.0
		if (size==0) {	
			for (int index=1;index<pn_array2.length;index++)
				pn_array2[index][flag]=0.0;
			return(null);
		}
		ExamplesStruct[] newExamples = new ExamplesStruct[size];

		// Loop through given examples array 
		int newIndex1=0;
		for (int index1=0;index1<examples.length;index1++) {
			// Example satisfies antecedent then copy to new examples set. copy
			// of PN array remains unchanged
			if (isSubset(antecedent,examples[index1].itemSet)) {
				newExamples[newIndex1] = new
						ExamplesStruct(examples[index1].itemSet,
								examples[index1].weighting);
				// Increment new index
				newIndex1++;
			}
			// Otherwise decrement counts in copy of PN array as appropriate
			// for records not included in the new example structure array
			else {
				long length = examples[index1].itemSet.length-1;
				for (int index2=0;index2<length;index2++) {
					long attribute = examples[index1].itemSet[index2];
					pn_array2[Math.toIntExact(attribute)][flag] = pn_array2[Math.toIntExact(attribute)][flag]-
							examples[index1].weighting;
				}
			}
		}

		// Return new example structure array reference
		return(newExamples);
	}

	/* REVISE WEIGHTINGS IN POSITIVE EXAMPLES ARRAY */
	/** Revises weightings in given positive example structure array and 
    adjust positice elements in the original PN array accordingly (not the
    copy). <P> Called whenever a new classification rule has been discovered 
    and added to the rule list.
    @param antecedent the given antecedent for the rule.
    @param examples the given positives example structure array.  */	

	protected void revisePosExDoSatRule(long[] antecedent,
			ExamplesStruct[] posExamples) {
		// Loop through given array
		for (int index1=0;index1<posExamples.length;index1++) {
			// Check if rule staisfies current record
			if (isSubset(antecedent,posExamples[index1].itemSet)) {
				// Get weighting
				double tempWeighting = posExamples[index1].weighting;
				// Reduce weighting associated with example
				posExamples[index1].weighting = tempWeighting*DECAY_FACROR;
				// Determine change in weighting
				double difference = tempWeighting-posExamples[index1].weighting;
				// Get length of record in positive example structure array.
				int length = posExamples[index1].itemSet.length-1;
				// Decrement weightings in PN array
				for (int index2=0;index2<length;index2++) {
					int attribute = Math.toIntExact(posExamples[index1].itemSet[index2]);
					pn_array[attribute][0]=pn_array[attribute][0]-difference;
				}
			}
		}
	}

	/* GET NUMBER EXAMPLES THAT DO SATISFY RULE. */

	/** Get the number of positive examples that satisfy the given rule
    antecedent for the given examples.
    @param antecedent the given antecedent
    @param examples the given example set.
    @return the number of example. */

	private int getNumExDoSatRule(long[] antecedent,
			ExamplesStruct[] examples) {
		// Determine number of positive examples that satisfy the given rule
		// antecedent
		int posWeigth=0;
		for(int index=0;index<examples.length;index++) {
			if (isSubset(antecedent,examples[index].itemSet)) posWeigth++;
		}

		// Return
		return(posWeigth);
	}

	/* COPY  EXAMPLES STRUCTURE ARRAY */

	/** Makes a copy of the given example structure array and returns the copy.
    @param examples the given example structure array
    @return copy of the given examples array. */

	protected ExamplesStruct[]
			copyExamplesArray(ExamplesStruct[] examples) {
		// Test for empty set
		if (examples==null) return(null);

		// Dimension new examples array
		ExamplesStruct[] newExamples = new
				ExamplesStruct[examples.length];

		// Loop through examples
		for (int index=0;index<examples.length;index++) {
			newExamples[index] = new
					ExamplesStruct(examples[index].itemSet,
							examples[index].weighting);
		}

		// End
		return(newExamples);
	}

	/* ------------------------------------------------------------ */
	/*                                                              */
	/*                           PN ARRAY                           */
	/*                                                              */
	/* ------------------------------------------------------------ */

	/* GENERATE PN ARRAY */

	/** Commences process of generating the PNarray for the given class.
    @param classification the given class of interest.  */

	protected void generatePNarray() {
		// Dimension PNarray
		pn_array = new double[numOneItemSets-numClasses+1][2];

		// Loop through positive examples
		for (int index=0;index<positiveExamples.length;index++) {
			generatePNarrayP(positiveExamples[index].itemSet,
					positiveExamples[index].weighting);
		}

		// Loop through negative examples
		for (int index=0;index<negativeExamples.length;index++) {
			generatePNarrayN(negativeExamples[index].itemSet,
					negativeExamples[index].weighting);
		}
	}

	/* GENERATE PN ARRAY POSITIVE */

	/** Generates positive part of PN array for the given record.
    @param record the given record in the input set. */

	protected void generatePNarrayP(long[] record, double weighting) {
		// Loop through record excluding class (last attribute).
		int length = record.length-1;
		for(int index=0;index<length;index++) {
			pn_array[Math.toIntExact(record[index])][0] =  pn_array[Math.toIntExact(record[index])][0]+
					weighting;
		}
	}

	/* GENERATE PN ARRAY NEGATIVE */

	/** Generates negative part of PN array for the given record.
    @param record the given record in the input set. */
	protected void generatePNarrayN(long[] record, double weighting) {

		// Loop through record excluding class (last attribute).
		int length = record.length-1;
		for(int index=0;index<length;index++) {
			pn_array[Math.toIntExact(record[index])][1] =  pn_array[Math.toIntExact(record[index])][1]+
					weighting;
		}
	}

	/* NO VALID GAINS IN PN ARRAY */

	/** Boolean method to check whether an attribute with a gain above the
    minimum threshold exists or not. <P> It is possible, if no attributes
    with gain above the specified minimum are discovered, that the current
    totalWeight for P will never be reduced to below the threshold.
    @return true if no attributes, false otherwise. */

	protected boolean noValidGainsinPNarray() {
		boolean noValidGains = true;
		double oldPosWeigth   = getWeightOfExamples(positiveExamples2);
		double oldNegWeigth   = getWeightOfExamples(negativeExamples2);

		// Loop through PN array
		for (int index=1;index<pn_array2.length;index++) {
			double newPosWeigth=pn_array2[index][0];
			double newNegWeigth=pn_array2[index][1];
			// Calculate gain and add to attribute array
			double gain = calculateGain(newPosWeigth,newNegWeigth,
					oldPosWeigth,oldNegWeigth);
			// Check gain
			if (gain>MIN_BEST_GAIN) noValidGains=false;
		}

		// Return
		return(noValidGains);
	}

	/* COMBINE PN ARRAY WEIGHTINGS */	
	/** Produces a new PN array made up of the positive weightings derived as a
    consequence of the previous generation of a rule for the current class, and
    the negative weightings derived from the original set of negative examples
    contained in the old PN array.
    @param pnArrayRevised the revised array containing the required positive
    weightings
    @param pnArrayOld the original PN array containing weightings for the
    original set of negative examples.
    @return the ne combined PN array. */

	/*protected double[][] combinePNarrayWeightings(double[][] pnArrayRevised,
    			double[][] pnArrayOld) {
	// Diemsion PNarray
        double[][] newPNarray = new double[pnArrayOld.length][2];

	// Loop through given PN array
	for(int index=0;index<pnArrayOld.length;index++) {
	    newPNarray[index][0]=pnArrayRevised[index][0];
	    newPNarray[index][1]=pnArrayOld[index][1];
	    }

	// Return
	return(newPNarray);
	}*/	

	/* ------------------------------------------------------------ */
	/*                                                              */
	/*                            WEIGHTING                         */
	/*                                                              */
	/* ------------------------------------------------------------ */

	/* GET TOTAL WEIGHTING */

	/** Calculates and returns the total weighting for the given set of
    examples (records);
    @param examples the given set of example records (positive or negative).
    @return the total weighting. */

	protected double getTotalWeighting(ExamplesStruct[] examples) {
		double total=0.0;

		// Loop through examples
		for(int index=0;index<examples.length;index++) {
			total=total+examples[index].weighting;
		}

		// Return
		return(total);
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
		System.out.println("File name           = " + fileName);
		System.out.println("Num. classes        = " + numClasses);
		System.out.println("K value             = " + K_VALUE);
		System.out.println("Min. best gain      = " + MIN_BEST_GAIN);
		System.out.println("Total Weight Factor = " + TOTAL_WEIGHT_FACTOR);
		System.out.println("Decay factor        = " + DECAY_FACROR);
		System.out.println();
	}
}

