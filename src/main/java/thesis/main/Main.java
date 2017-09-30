package thesis.main;

import thesis.engine.FileProcessingEngine;
import thesis.engine.PreProcessingEngine;
import weka.core.Instances;

import java.util.Date;

public class Main {

    public static void main(String[] args) {
        Instances dataSet = FileProcessingEngine.readDataSetFromFile("data/ML_Data_original.arff");
        System.out.println("Data loaded from file successfully");

        PreProcessingEngine preProcessingEngine = new PreProcessingEngine(dataSet);
        preProcessingEngine.preProcess("rating",
                new String[] {"submit_date", "recommendation_score", "user_id", "tracker_id", "_id", "email", "created_at"},
                new String[] {"gsm_rssi", "pos_uncertainty", "no_cell_locates", "no_of_sat", "days_in_use",
                              "cmd_success_rate", "cmd_terminated_rate", "cmd_cancelled_rate", "cmd_delay_to_confirmed",
                              "cmd_delay_to_pos_any", "cmd_delay_to_pos_new"}, 0.0, 1.0);
        System.out.println("Data preprocessed successfully");

        FileProcessingEngine.writeDataSetToFile(preProcessingEngine.getDataSet(), "data/ML_Data_preprocessed_" + new Date().toString() + ".arff");
        System.out.println("Data written to file successfully");
    }
}
