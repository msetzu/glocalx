/* -------------------------------------------------------------------------- */
/*                                                                            */
/*                      ASSOCIATION RULE DATA MINING                          */
/*                                                                            */
/*                            Frans Coenen                                    */
/*                                                                            */
/*                        Wednesday 9 January 2003                            */
/*        (revised 21/1/2003, 14/2/2003, 2/5/2003, 2/7/2003, 3/2/2004)        */
/*                                                                            */
/*                    Department of Computer Science                          */
/*                     The University of Liverpool                            */
/*                                                                            */ 
/* -------------------------------------------------------------------------- */

// Java packages
import java.io.*;
import java.util.*;

// Java GUI packages
import javax.swing.*;

/** Set of utilities to support various Association Rule Mining (ARM) 
algorithms included in the LUCS-KDD suite of ARM programs. 
@author Frans Coenen
@version 2 July 2003 */

public class AssocRuleMining {

	/* ------ FIELDS ------ */

	// Data structures

	/** 2-D aray to hold input data from data file */
	protected long[][] dataArray = null;


	// Command line arguments with default values and associated fields

	/** Command line argument for data file name. */	
	protected String  fileName = null;
	/** Number of classes in input data set (input by the user). */
	protected int numClasses   = 0;	

	// Flags

	/** Error flag used when checking command line arguments (default = 
    <TT>true</TT>). */
	protected boolean errorFlag  = true;
	/** Input format OK flag( default = <TT>true</TT>). */
	protected boolean inputFormatOkFlag = true;

	// Other fields

	/** Number of columns. */	
	protected int     numCols    = 0;
	/** Number of rows. */
	protected int     numRows    = 0;
	/** The number of one itemsets (singletons). */
	protected int numOneItemSets = 0;	
	/** The input stream. */
	protected BufferedReader fileInput;
	/** The file path */
	protected File filePath = null;	

	/* ------ CONSTRUCTORS ------ */

	/** Processes command line arguments */		
	public AssocRuleMining(String[] args) {

		// Process command line arguments
		for(int index=0;index<args.length;index++) 
			idArgument(args[index]);

		// If command line arguments read successfully (errorFlag set to "true")
		// check validity of arguments
		if (errorFlag) CheckInputArguments();
		else outputMenu();
	}

	/** Default constructor */

	public AssocRuleMining() {
		;
	}

	/* ------ METHODS ------ */	

	/* ---------------------------------------------------------------- */
	/*                                                                  */
	/*                        COMMAND LINE ARGUMENTS                    */
	/*                                                                  */
	/* ---------------------------------------------------------------- */

	/* IDENTIFY ARGUMENT */
	/** Identifies nature of individual command line arguments:
    -N = number of classes, -F = file name, -S = support. */

	protected void idArgument(String argument) {

		if (argument.charAt(0) == '-') {
			char flag = argument.charAt(1);
			argument = argument.substring(2,argument.length());
			switch (flag) {
			case 'F':
				fileName = argument;
				break;
			case 'N':
				numClasses =  Integer.parseInt(argument);  
				break;
			default:
				System.out.println("INPUT ERROR: Unrecognised command " +
						"line  argument -" + flag + argument);
				errorFlag = false;  
			}     
		}
		else {
			System.out.println("INPUT ERROR: All command line arguments " +
					"must commence with a '-' character (" + 
					argument + ")");
			errorFlag = false;
		}
	}

	/* CHECK INPUT ARGUMENTS */
	/** Invokes methods to check values associate with command line
    arguments */

	protected void CheckInputArguments() {	 
		/* STUB */
	}

	/* CHECK FILE NAME */
	/** Checks if data file name provided, if not <TT>errorFlag</TT> set 
    to <TT>false</TT>. */	

	protected void checkFileName() {
		if (fileName == null) {
			System.out.println("INPUT ERROR: Must specify file name (-F)");
			errorFlag = false;		
		}		
	}

	/* ---------------------------------------------------------------- */
	/*                                                                  */
	/*                     READ INPUT DATA FROM FILE                    */
	/*                                                                  */
	/* ---------------------------------------------------------------- */

	/* READ FILE */
	/** Reads input data from file specified in command line argument (GUI 
    version also exists). <P>Proceeds as follows:
    <OL>
    <LI>Gets number of lines in file, checking format of each line (space 
    separated integers), if incorrectly formatted line found 
    <TT>inputFormatOkFlag</TT> set to <TT>false</TT>.
    <LI>Dimensions input array.
    <LI>Reads data
    </OL> */

	protected void readFile() {
		try {
			// Dimension data structure
			inputFormatOkFlag = true;
			numRows = getNumberOfLines(fileName);
			if (inputFormatOkFlag) {
				dataArray = new long[numRows][];	
				// Read file	
				//System.out.println("Reading input file: " + fileName); 
				readInputDataSet();
			} else System.out.println("Error reading file: " + fileName + "\n");
		}
		catch(IOException ioException) { 
			System.out.println("Error reading File");
			closeFile();
			System.exit(1);
		}	 
	}    

	/* GET NUMBER OF LINES */

	/** Gets number of lines/records in input file and checks format of each 
    line. 
    @param nameOfFile the filename of the file to be opened.
    @return the number pf rows in the given file. */
	
	protected int getNumberOfLines(String nameOfFile) throws IOException {
		int counter = 0;

		// Open the file
		if (filePath==null) openFileName(nameOfFile);
		else openFilePath();

		// Loop through file incrementing counter
		// get first row.
		String line = fileInput.readLine();	
		while (line != null) {
			checkLine(counter+1,line);
			StringTokenizer dataLine = new StringTokenizer(line);
			int numberOfTokens = dataLine.countTokens();
			if (numberOfTokens == 0) break;
			counter++;	 
			line = fileInput.readLine();
		}

		// Close file and return
		closeFile();
		return(counter);
	}

	/* CHECK LINE */

	/** Check whether given line from input file is of appropriate format
    (space separated integers), if incorrectly formatted line found 
    <TT>inputFormatOkFlag</TT> set to <TT>false</TT>. 
    @param counter the line number in the input file.
    @param str the current line from the input file. */	

	protected void checkLine(int counter, String str) {

		for (int index=0;index <str.length();index++) {
			if (!Character.isDigit(str.charAt(index)) &&
					!Character.isWhitespace(str.charAt(index))) {
				JOptionPane.showMessageDialog(null,"FILE INPUT ERROR:\n" +
						"charcater on line " + counter + 
						" is not a digit or white space");	        
				inputFormatOkFlag = false;
				break;
			}
		}
	}

	/* READ INPUT DATA SET */    
	/** Reads input data from file specified in command line argument. */

	public void readInputDataSet() throws IOException {  
		int rowIndex=0;

		// Open the file
		if (filePath==null) openFileName(fileName);
		else openFilePath();

		// get first row.
		String line = fileInput.readLine();	
		while (line != null) {
			StringTokenizer dataLine = new StringTokenizer(line);
			int numberOfTokens = dataLine.countTokens();
			if (numberOfTokens == 0) break;
			// Convert input string to a sequence of long integers
			long[] code = binConversion(dataLine,numberOfTokens);
			// Check for "null" input
			if (code != null) {
				// Dimension row in 2-D dataArray
				int codeLength = code.length;
				dataArray[rowIndex] = new long[codeLength];
				// Assign to elements in row
				for (int colIndex=0;colIndex<codeLength;colIndex++)
					dataArray[rowIndex][colIndex] = code[colIndex];
			} else dataArray[rowIndex]= null;
			// Increment first index in 2-D data array
			rowIndex++;
			// get next line
			line = fileInput.readLine();
		}

		// Close file
		closeFile();
	}	

	/* CHECK DATASET ORDERING */ 
	/** Checks that data set is ordered correctly. */

	protected boolean checkOrdering() {
		boolean result = true; 

		// Loop through input data
		for(int index=0;index<dataArray.length;index++) {
			if (!checkLineOrdering(index+1,dataArray[index])) result=false;
		}

		// Return 
		return(result);
	}

	/* CHECK LINE ORDERING */
	/** Checks whether a given line in the input data is in numeric sequence.
    @param lineNum the line number.
    @param itemSet the item set represented by the line
    @return true if OK and false otherwise. */

	private boolean checkLineOrdering(int lineNum, long[] itemSet) {
		for (int index=0;index<itemSet.length-1;index++) {
			if (itemSet[index] > itemSet[index+1]) {
				JOptionPane.showMessageDialog(null,"FILE FORMAT ERROR:\n" +
						"Attribute data in line " + lineNum + 
						" not in numeric order");
				return(false);
			}
		}    

		// Default return
		return(true);
	}

	/* COUNT NUMBER OF COLUMNS */
	/** Counts number of columns represented by input data. */

	protected void countNumCols() {
		int maxAttribute=0;

		// Loop through data array	
		for(int index=0;index<dataArray.length;index++) {
			int lastIndex = dataArray[index].length-1;
			if (dataArray[index][lastIndex] > maxAttribute)
				maxAttribute = Math.toIntExact(dataArray[index][lastIndex]);
		}

		numCols        = maxAttribute;
		numOneItemSets = numCols; 	// default value only
	}	

	/* OPEN FILE NAME */    
	/** Opens file using fileName (instance field). 
    @param nameOfFile the filename of the file to be opened. */

	protected void openFileName(String nameOfFile) {
		try {
			// Open file
			FileReader file = new FileReader(nameOfFile);
			fileInput = new BufferedReader(file);
		}
		catch(IOException ioException) {
			System.err.println("Error Opening File");
		}
	}

	/* OPEN FILE PATH */
	/** Opens file using filePath (instance field). */

	private void openFilePath() {
		try {
			// Open file
			FileReader file = new FileReader(filePath);
			fileInput = new BufferedReader(file);
		}
		catch(IOException ioException) {
			System.err.println("Error Opening File");
		}
	}

	/* CLOSE FILE */    
	/** Close file fileName (instance field). */

	protected void closeFile() {
		if (fileInput != null) {
			try {
				fileInput.close();
			}
			catch (IOException ioException) {
				System.err.println("Error Closing File");
			}
		}
	}

	/* BINARY CONVERSION. */

	/** Produce an item set (array of elements) from input line.
    @param dataLine row from the input data file
    @param numberOfTokens number of items in row
    @return 1-D array of long integers representing attributes in input
    row */

	protected long[] binConversion(StringTokenizer dataLine, 
			int numberOfTokens) {
		long number;
		long[] newItemSet = null;

		// Load array

		for (int tokenCounter=0;tokenCounter < numberOfTokens;tokenCounter++) {
			number = new Long(dataLine.nextToken()).longValue();
			newItemSet = realloc1(newItemSet,number);
		}

		// Return itemSet	

		return(newItemSet);
	}

	/* ----------------------------------------------- */
	/*                                                 */
	/*        ITEM SET INSERT AND ADD METHODS          */
	/*                                                 */
	/* ----------------------------------------------- */

	/* REALLOC INSERT */

	/** Resizes given item set so that its length is increased by one
    and new element inserted.
    @param oldItemSet the original item set
    @param newElement the new element/attribute to be inserted
    @return the combined item set */

	protected long[] reallocInsert(long[] oldItemSet, long newElement) {	

		// No old item set

		if (oldItemSet == null) {
			long[] newItemSet = {newElement};
			return(newItemSet);
		}

		// Otherwise create new item set with length one greater than old
		// item set

		int oldItemSetLength = oldItemSet.length;
		long[] newItemSet = new long[oldItemSetLength+1];

		// Loop

		int index1;	
		for (index1=0;index1 < oldItemSetLength;index1++) {
			if (newElement < oldItemSet[index1]) {
				newItemSet[index1] = newElement;	
				// Add rest	
				for(int index2 = index1+1;index2<newItemSet.length;index2++)
					newItemSet[index2] = oldItemSet[index2-1];
				return(newItemSet);
			}
			else newItemSet[index1] = oldItemSet[index1];
		}

		// Add to end

		newItemSet[newItemSet.length-1] = newElement;

		// Return new item set

		return(newItemSet);
	}

	/* REALLOC 1 */

	/** Resizes given item set so that its length is increased by one
    and appends new element (identical to append method)
    @param oldItemSet the original item set
    @param newElement the new element/attribute to be appended
    @return the combined item set */

	protected long[] realloc1(long[] oldItemSet, long newElement) {

		// No old item set

		if (oldItemSet == null) {
			long[] newItemSet = {newElement};
			return(newItemSet);
		}

		// Otherwise create new item set with length one greater than old
		// item set

		int oldItemSetLength = oldItemSet.length;
		long[] newItemSet = new long[oldItemSetLength+1];

		// Loop

		int index;
		for (index=0;index < oldItemSetLength;index++)
			newItemSet[index] = oldItemSet[index];
		newItemSet[index] = newElement;

		// Return new item set

		return(newItemSet);
	}

	/* ---------------------------------------------------------------- */
	/*                                                                  */
	/*              METHODS TO RETURN SUBSETS OF ITEMSETS               */
	/*                                                                  */
	/* ---------------------------------------------------------------- */

	/* GET LAST ELEMENT */

	/** Gets thelast element in the given item set, or '0' if the itemset is
    empty.
    @param itemSet the given item set.
    @return the last element. */

	protected long getLastElement(long[] itemSet) {
		// Check for empty item set
		if (itemSet == null) return(0);
		// Otherwise return last element
		return(itemSet[itemSet.length-1]);
	}  	

	/* ----------------------------------------------------- */
	/*                                                       */
	/*             BOOLEAN ITEM SET METHODS ETC.             */
	/*                                                       */
	/* ----------------------------------------------------- */  	

	/* EQUALITY CHECK */

	/** Checks whether two item sets are the same.
    @param itemSet1 the first item set.
    @param itemSet2 the second item set to be compared with first.
    @return true if itemSet1 is equal to itemSet2, and false otherwise. */

	protected boolean isEqual(long[] itemSet1, long[] itemSet2) {

		// If no itemSet2 (i.e. itemSet2 is null return false)

		if (itemSet2 == null) return(false);

		// Compare sizes, if not same length they cannot be equal.

		int length1 = itemSet1.length;
		int length2 = itemSet2.length;
		if (length1 != length2) return(false);

		// Same size compare elements

		for (int index=0;index < length1;index++) {
			if (itemSet1[index] != itemSet2[index]) return(false);
		}

		// itemSet the same.

		return(true);
	}

	/* ----------------------------------------------------- */
	/*                                                       */
	/*             BOOLEAN ITEM SET METHODS ETC.             */
	/*                                                       */
	/* ----------------------------------------------------- */

	/* SUBSET CHECK */

	/** Checks whether one item set is subset of a second item set.
    @param itemSet1 the first item set.
    @param itemSet2 the second item set to be compared with first.
    @return true if itemSet1 is a subset of itemSet2, and false otherwise.
	 */

	protected boolean isSubset(long[] itemSet1, long[] itemSet2) {
		// Check for empty itemsets
		if (itemSet1==null) return(true);
		if (itemSet2==null) return(false);

		// Loop through itemSet1
		for(int index1=0;index1<itemSet1.length;index1++) {
			if (notMemberOf(itemSet1[index1],itemSet2)) return(false);
		}

		// itemSet1 is a subset of itemSet2
		return(true);
	}

	/* NOT MEMBER OF */

	/** Checks whether a particular element/attribute identified by a
    column number is not a member of the given item set.
    @param number the attribute identifier (column number).
    @param itemSet the given item set.
    @return true if first argument is not a member of itemSet, and false
    otherwise */

	protected boolean notMemberOf(long number, long[] itemSet) {

		// Loop through itemSet

		for(int index=0;index<itemSet.length;index++) {
			if (number < itemSet[index]) return(true);
			if (number == itemSet[index]) return(false);
		}

		// Got to the end of itemSet and found nothing, return false

		return(true);
	}	

	/* ---------------------------------------------------------------- */
	/*                                                                  */
	/*                            MISCELANEOUS                          */
	/*                                                                  */
	/* ---------------------------------------------------------------- */

	/* COPY ITEM SET */

	/** Makes a copy of a given itemSet.
    @param itemSet the given item set.
    @return copy of given item set. */

	protected long[] copyItemSet(long[] itemSet) {

		// Check whether there is a itemSet to copy
		if (itemSet == null) return(null);

		// Do copy and return
		long[] newItemSet = new long[itemSet.length];
		for(int index=0;index<itemSet.length;index++) {
			newItemSet[index] = itemSet[index];
		}

		// Return
		return(newItemSet);
	}

	/* ------------------------------------------------- */
	/*                                                   */
	/*                   OUTPUT METHODS                  */
	/*                                                   */
	/* ------------------------------------------------- */

	/* ----------------- */	
	/* OUTPUT DATA TABLE */
	/* ----------------- */
	/** Outputs stored input data set; initially read from input data file, but
    may be reordered or pruned if desired by a particular application. */

	public void outputDataArray() {
		System.out.println("DATA SET\n" + "--------");

		// Loop through data array
		for(int index=0;index<dataArray.length;index++) {
			outputItemSet(dataArray[index]);
			System.out.println();
		}
	}

	/* -------------- */
	/* OUTPUT ITEMSET */
	/* -------------- */
	/** Outputs a given item set.
    @param itemSet the given item set. */

	protected void outputItemSet(long[] itemSet) {

		// Loop through item set elements

		if (itemSet == null) System.out.print(" ");
		else {
			int counter = 0;
			for (int index=0; index<itemSet.length; index++) {
				if (counter == 0) {
					counter++;
					//System.out.print(" {");
				}
				else System.out.print(",");
				System.out.print(itemSet[index]);
			}
			//System.out.print("} ");
		}
	}

	/* ---------------------- */		
	/* OUTPUT DATA ARRAY SIZE */
	/* ---------------------- */
	/** Outputs size (number of records and number of elements) of stored
    input data set read from input data file. */

	public void outputDataArraySize() {
//		int numRecords = 0;
		int numElements = 0;

		// Loop through data array

		for (int index=0;index<dataArray.length;index++) {
			if (dataArray[index] != null) {
//				numRecords++;
				numElements = numElements+dataArray[index].length;
			}
		}

		// Output
		//System.out.println("Number of columns  = " + numCols);
		//System.out.println("Number of records  = " + numRecords);
		//System.out.println("Number of elements = " + numElements);
		//double density = (double) numElements/ (numCols*numRecords);
		//System.out.println("Data set density   = " + twoDecPlaces(density) +						"%");
	}

	/* ----------- */
	/* OUTPUT MENU */
	/* ----------- */
	/** Outputs menu for command line arguments. */

	protected void outputMenu() {
		System.out.println();
		System.out.println("-F  = File name");	
		System.out.println("-N  = Number of classes (Optional)");
		System.out.println();

		// Exit

		System.exit(1);
	}
	/* --------------- */
	/* OUTPUT SETTINGS */
	/* --------------- */
	/** Outputs command line values provided by user. */

	protected void outputSettings() {
		/* STUB */
	}

	/* --------------------------------- */
	/*                                   */
	/*        DIAGNOSTIC OUTPUT          */
	/*                                   */
	/* --------------------------------- */

	/* OUTPUT DURATION */
	/** Outputs difference between two given times.
    @param time1 the first time.
    @param time2 the second time.
    @return duration. */

	public double outputDuration(double time1, double time2) {
		double duration = (time2-time1)/1000;
		System.out.println("Generation time = " + twoDecPlaces(duration) +
				" seconds (" + twoDecPlaces(duration/60) + " mins)");

		// Return
		return(duration);
	}

	/* -------------------------------- */
	/*                                  */
	/*        OUTPUT UTILITIES          */
	/*                                  */
	/* -------------------------------- */

	/* TWO DECIMAL PLACES */

	/** Converts given real number to real number rounded up to two decimal
    places.
    @param number the given number.
    @return the number to two decimal places. */

	protected double twoDecPlaces(double number) {
		int numInt = (int) ((number+0.005)*100.0);
		number = ((double) numInt)/100.0;
		return(number);
	}	
}

