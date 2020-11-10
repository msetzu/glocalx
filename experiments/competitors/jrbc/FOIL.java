import java.io.*;

public class FOIL {

	public static void main(String[] args) throws IOException {

		// Create instance of class ClassificationFOIL	
		FOIL_CARgen foil = new FOIL_CARgen(args);

		foil.inputDataSet();

		foil.fit();

		// Output
		foil.getCurrentRuleListObject().outputRules();

		// End
		System.exit(0);
	}
}
