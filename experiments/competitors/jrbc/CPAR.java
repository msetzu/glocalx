import java.io.*;

public class CPAR {

	public static void main(String[] args) throws IOException {

		// Create instance of class ClassificationFOIL	
		CPAR_CARgen cpar = new CPAR_CARgen(args);

		cpar.inputDataSet();

		cpar.fit();

		// Output
		cpar.getCurrentRuleListObject().outputRules();

		// End
		System.exit(0);
	}
}