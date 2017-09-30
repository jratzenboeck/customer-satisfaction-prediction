package thesis.engine;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;

public class FileProcessingEngine {

    public static Instances readDataSetFromFile(String filename) {
        try {
            DataSource source = new DataSource(filename);
            return source.getDataSet();
        } catch (Exception e) {
            throw new RuntimeException("Data set could not be loaded from passed filename", e);
        }
    }

    public static void writeDataSetToFile(Instances dataSet, String filename) {
        try {
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(dataSet);
            arffSaver.setFile(new File(filename));
            arffSaver.writeBatch();
        } catch (Exception e) {
            throw new RuntimeException("Data set could not be written to file", e);
        }
    }
}
