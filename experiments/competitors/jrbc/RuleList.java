/* -------------------------------------------------------------------------- */
/*                                                                            */
/*                              RULE LIST                                     */
/*                                                                            */
/*                            Frans Coenen                                    */
/*                    Department of Computer Science                          */
/*                     The University of Liverpool                            */
/*                                                                            */
/*               Tuesday 2 March 2004, Revised 11 Januray 2005                */
/*                                                                            */
/* Bug reports: Thanks to Jian (Ella) Chen, Dept Comp. Sci. School of         */
/*     Information Sci. and Tech. Sun Yat-sen University, China.              */
/*                                                                            */ 
/* -------------------------------------------------------------------------- */

/* Class structure

AssocRuleMining
      |
      +-- RuleList			*/

// Java packages
import java.io.*;
import java.util.*;

// Java GUI packages
import javax.swing.*;

/** Set of utilities to support various Association Rule Mining (ARM) 
algorithms included in the LUCS-KDD suite of ARM programs. 
@author Frans Coenen
@version 2 March 2004 */

public class RuleList extends AssocRuleMining {

    /* ------ FIELDS ------ */
	
    // --- Data structures ---
    
    /** Rule node in linked list of rules (either ARs or CRs). */

    protected class RuleNode {
    	/** Antecedent of AR. */
	protected long[] antecedent;
	/** Consequent of AR. */
	protected long[] consequent;
	/** The Laplace accuracy associate with the rule represented by this
	node. */
	double laplaceAccuracy=0.0;
	/** Link to next node */
	RuleNode next = null;
	
	/** Three argument constructor
	@param antecedent the antecedent (LHS) of the AR.
    	@param consequent the consequent (RHS) of the AR.
    	@param accuracy the associated Laplace accuracy value. */
	
	private RuleNode(long[] ante, long[]cons, double accuracy) {
	    antecedent      = ante;
	    consequent      = cons;
	    laplaceAccuracy = accuracy;
	    }
	}
	
    /** The reference to start of the rule list. */
    protected RuleNode startRulelist = null;

    /* ------ CONSTRUCTORS ------ */

    /** Default constructor to create an instance of the class RuleList  */
    	
    public RuleList() {;}
	
    /* ------ METHODS ------ */	

    /* -------------------------------------------------------------- */
    /*                                                                */
    /*     RULE LINKED LIST ORDERED ACCORDING TO LAPLACE ACCURACY     */
    /*                                                                */
    /* -------------------------------------------------------------- */

    /* Methods for inserting rules into a linked list of rules ordered
    according to laplace accuracy (most accurate rule first). Each rule
    described in terms of 3 fields: 1) Antecedent (an item set), 2) a
    consequent (an item set), 3) a Laplace accuracy value (double).  */

    /* INSERT (ASSOCIATION/CLASSIFICATION) RULE INTO RULE LINKED LIST (ORDERED
    ACCORDING LAPLACE ACCURACY). */

    /** Inserts an (association/classification) rule into the linked list of
    rules pointed at by <TT>startRulelist</TT>. <P> The list is ordered so that
    rules with highest confidence are listed first. If two rules have the same
    confidence the new rule will be placed after the existing rule. Thus, if
    using an Apriori approach to generating rules, more general rules will
    appear first in the list with more specific rules (i.e. rules with a larger
    antecedent) appearing later as the more general rules will be generated
    first.
    @param antecedent the antecedent (LHS) of the rule.
    @param consequent the consequent (RHS) of the rule.
    @param laplaceAccuracy the associated Laplace accuracy value for the
    rule.*/

    protected void insertRuleintoRulelist(
		long[] antecedent, long[] consequent, double laplaceAccuracy) {

	// Create new node
	RuleNode newNode = new RuleNode(antecedent, consequent, laplaceAccuracy);
	
	// Empty list situation
	if (startRulelist == null) {
	    startRulelist = newNode;
	    return;
	    }
		
	// Add new node to start	
	if (laplaceAccuracy > startRulelist.laplaceAccuracy) {
	    newNode.next = startRulelist;
	    startRulelist  = newNode;
	    return;
	    }
	
	// Add new node to middle
	RuleNode markerNode = startRulelist;
	RuleNode linkRuleNode = startRulelist.next;
	while (linkRuleNode != null) {
	    if (laplaceAccuracy > linkRuleNode.laplaceAccuracy) {
	        markerNode.next = newNode;
		newNode.next    = linkRuleNode;
		return;
		}
	    markerNode = linkRuleNode;
	    linkRuleNode = linkRuleNode.next;	
	    }
	
	// Add new node to end
	markerNode.next = newNode;
	}
	
    /* -------------------------------------------------------- */
    /*                                                          */
    /*              LAPLACE EXPECTED ERROR ESTIMATE             */
    /*                                                          */
    /* -------------------------------------------------------- */
	
    /* CALCULATE LAPLACE ACCURACY */

    /** Determines Laplace expected error estimates (accuracies).<P> Note:
    only appropriate for classification rule generators, used by FOIL, PRM and
    CPAR. Calculated as follows:
    <PRE>
    Nc   = Number of records in training set that include all the attributes
    	   for the given rule antecedent + consequent, i.e. the total support
	   for the rule.
    Ntot = Number of records in training set that include all the
           attributes in the rule's antecedent, i.e. support for antecedent.
    accuracy = (Nc+1)/(Ntot+numberOfClasses)
    </PRE>
    @param antecedent the antecedent of the given rule.
    @param consequent the consequent of the given rule.
    @return the Laplace accuracy. */

    protected double getLaplaceAccuracy(long[] antecedent, long consequent) {
    	int totalCounter=0;	// Ntot
		int classCounter=0;	// Nc
	
		// Get number of records in training set that satisfy rule antecedent
		// only (Ntot), and the size of the sunset of the identified records that
		// also include the rule consequent (Nc)
		for(int index=0;index<dataArray.length;index++) {
	    	if (isSubset(antecedent, dataArray[index])) {
	        	int lastIndex = dataArray[index].length-1;
	        	if (consequent==dataArray[index][lastIndex]) classCounter++;
	        	totalCounter++;
	        }
	    }
	
		// Return
		double accuracy = (double) (classCounter+1)/(double) (totalCounter+numClasses);
		//double coverage = (double) 
		return(accuracy);
	}  	
	
    /* ------------------------------------------------------------- */
    /*                 CLASSIFIER  (BEST K AVERAGE)                  */
    /* ------------------------------------------------------------- */

    /* CLASSIFY RECORD (BEST K AVERAGE) */
    /** Selects the best rule in a rule list according to the average expected
    Laplace accuracy. <P> Used in connection with FOIL, PRM and CPAR. Operates
    as follows:
    1) Obtain all rules whose antecedent is a subset of the given record.
    2) Select the best K rules for each class (according to their Laplace
    accuracy).
    3) Determine the average expected accuracy over the selected rules for each
    class,
    4) Select the class with the best average expected accuracy.
    @param kValue the maximum number of rules to be considered to classify the
    given record.
    @param classification the possible classers.
    @param itemset the record to be classified.
    @return the class label.		*/

    protected long classifyRecordBestKaverage(int kValue,
    				long[] classification, long[] itemSet) {	
        RuleNode linkRef = startRulelist;
	RuleNode tempRulelist = startRulelist;	
	startRulelist=null;
	
	// Obtain rules that satisfy record (iremSet)
	obtainallRulesForRecord(linkRef,itemSet);
       
	// Keep only best K rules for each class.
	keepBestKrulesPerClass(kValue,classification);

	// Determine average expected accuracies
	double[] averages = getAverageAccuracies(Math.toIntExact(classification[0])); 
	
	// Select best average and return
        long classLabel =
		(long) selectClassWithBestAverage(averages,Math.toIntExact(classification[0]));
	
	// Reset global rule list reference
	startRulelist=tempRulelist;
	
	// Return class
	return(classLabel);
	}

    /* KEEP BEST K RULES PER CLASS */
    /** Keep only best K rules for each class. <P> Note rules are stored
    according to Laplace accuracy (most accurate rule first), this is the
    measure to be maximised to identify "best" rules.
    @param kValue the maximum number of rules to be kept per class.
    @param classification the possible classes. */

    private void keepBestKrulesPerClass(int kValue, long[] classification) {
	// Loop through classification array
	for (int index=0;index<classification.length;index++) {
	    int counter=0;
	    RuleNode linkRef = startRulelist;
	    RuleNode markerRef = null;
	    // Loop through linked list of selected rules that satisfy the
	    // given record
	    while (linkRef!=null) {	
	        // Rule consequent matches current class    	
		if (classification[index]==linkRef.consequent[0]) {
		    // If counter less than maximum increment counter and 
		    // marker reference
		    if (counter<kValue) {
		        counter++;
			markerRef=linkRef;
			}
		    // Otherwise remove rule (and maintain marker reference).
		    else {
		        if (markerRef== null) startRulelist=linkRef.next;
			else markerRef.next=linkRef.next;
			}
		    }
		// No match, increment marker 
		else markerRef=linkRef;
		// Next rule
		linkRef=linkRef.next;
		}
	    }
	}

    /* GET AVERAGE ACCURACIES */
    /** Returns average accuracies for each class
    @param decrement the label for the first class in the classification array
    which is used here to relate class labels to indexes.
    @return the accuracy array describing accuracies for each class. */
    	
    private double[] getAverageAccuracies(int decrement) {
		double[] accuracies = new double[numClasses];
		int[] totals = new int[numClasses];
	
		// Loop through linked list of best selected rules that satisfy the
		// given record and determine total accuracies for each class
		RuleNode linkRef = startRulelist;
		while (linkRef!=null) {	
	    	int index = Math.toIntExact(linkRef.consequent[0]-decrement);
	    	accuracies[index] = accuracies[index]+linkRef.laplaceAccuracy;
	    	totals[index]++;
	    	linkRef=linkRef.next;
	    }
	
		// Determine averages
		for(int index=0;index<accuracies.length;index++) {
	    	if (accuracies[index]!=0)
	    		accuracies[index] = accuracies[index]/totals[index];
	    }
	
		// return
		return(accuracies);
	}

    /* SELECT CLASS WITH BEST AVERAGE */
    /** Returns the class label with the best average accuracy associated with
    it.
    @param increment the label for the first class in the classification array
    which is used here to relate class labels to indexes.
    @param the given list of averages
    @return the class label. */	
	
    private int selectClassWithBestAverage(double[] averages, int increment) {
	int bestIndex=0;
	double bestAverage=averages[bestIndex];
	
	// Loop through array of averages
	int index=1;
	for ( ;index<averages.length;index++) {
	    if (averages[index]>bestAverage) {
	        bestIndex=index;
		bestAverage=averages[index];
		}
	    }
	
	// Return class
        return(bestIndex+increment);
	}
	
    /* ------------------------------------------------------------- */
    /*                 CLASSIFIER  (UTILITY METHODS)                 */
    /* ------------------------------------------------------------- */
	
    /* OBTAIN RULES FOR RECORD */
    /** Places all rules that satisfy the given record in a rule linked list
    pointed at by startRulelist field, in the order that rules are presented.
    <P> Used in Best K Average (CPAR)
    algorithm.
    @param linkref The reference to the start of the existing list of rules.
    @param itemset the record to be classified.	*/

    private void obtainallRulesForRecord(RuleNode linkRef, long[] itemSet) {
	RuleNode newStartRef = null;
	RuleNode markerRef   = null;
	
	// Loop through linked list of existing rules
	while (linkRef!=null) {
	    // If rule satisfies record add to new rule list
	    if (isSubset(linkRef.antecedent,itemSet)) {
	   	RuleNode newNode = new RuleNode(linkRef.antecedent,
				linkRef.consequent,linkRef.laplaceAccuracy);
	   	if (newStartRef==null) newStartRef=newNode;
		else markerRef.next=newNode;
		markerRef=newNode;
		}
	    linkRef=linkRef.next;
	    }
	
	// Set rule list
	startRulelist = newStartRef;
	}	
	
    /* ----------------------------------- */
    /*                                     */
    /*              GET METHODS            */
    /*                                     */
    /* ----------------------------------- */
	
    /* GET NUMBER OF RULES */

    /**  Returns the number of generated rules (usually used in
    conjunction with classification rule mining algorithms rather than ARM
    algorithms).
    @return the number of CRs. */

    protected int getNumCRs() {
        int number = 0;
        RuleNode linkRuleNode = startRulelist;
	
	// Loop through linked list
	while (linkRuleNode != null) {
	    number++;
	    linkRuleNode = linkRuleNode.next;
	    }
	
	// Return
	return(number);
	}
	
    /* ----------------------------------- */
    /*                                     */
    /*              SET METHODS            */
    /*                                     */
    /* ----------------------------------- */
	
    /* SET NUMBER OF CLASSES */

    /** Sets number of rows field. */

    protected void setNumClasses(int numC) {
        numClasses=numC;
	}

    /* SET DATA ARRAT */

    /** Set 2-D "long" data array reference. */

    protected void setDataArray(long[][] dArray) {
        dataArray=dArray;
	}
	
    /* ------------------------------ */
    /*                                */
    /*              OUTPUT            */
    /*                                */
    /* ------------------------------ */

    /* OUTPUT RULE LINKED LIST */
    /** Outputs contents of rule linked list (if any) */

    public void outputRules() {
        outputRules(startRulelist);
	}
	
    /** Outputs given rule list.
    @param ruleList the given rule list. */

    private void outputRules(RuleNode ruleList) {
	//System.out.println("CLASSIFIER\n----------\b");
	
	// Check for empty rule list
	if (ruleList==null) System.out.println("No rules generated!");
	
	// Loop through rule list
	int number = 1;
        RuleNode linkRuleNode = ruleList;
	while (linkRuleNode != null) {
	    //System.out.print("(" + number + ") ");
	    outputRule(linkRuleNode);
        System.out.println(linkRuleNode.laplaceAccuracy);
	    number++;
	    linkRuleNode = linkRuleNode.next;
	    }
	}

    /** Outputs a rule.
    @param rule the rule to be output. */

    private void outputRule(RuleNode rule) {
		outputItemSet(rule.consequent);
		System.out.print(';');
        outputItemSet(rule.antecedent);
		System.out.print(';');
	}

    /* OUTPUT NUMBER OF RULES */
    
    /** Outputs number of generated rules (ARs or CARS). */
    
    public void outputNumRules() {
        System.out.println("Number of rules         = " + getNumCRs());
	}
    }

