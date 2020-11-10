import java.io.*;

public class PRM {

	public static void main(String[] args) throws IOException {

		// Create instance of class ClassificationFOIL	
		PRM_CARgen prm = new PRM_CARgen(args);

		prm.inputDataSet();

		prm.fit();

		// Output
		prm.getCurrentRuleListObject().outputRules();

		// End
		System.exit(0);
	}
}
