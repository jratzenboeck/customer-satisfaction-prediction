package thesis.engine;

import org.junit.Assert;
import org.junit.Test;

public class FileProcessingEngineTests {

    private static final int EXPECTED_NUM_INSTANCES = 3;

    @Test
    public void testReadDataSetFromFile() {
        Assert.assertEquals(EXPECTED_NUM_INSTANCES,
                FileProcessingEngine.readDataSetFromFile("data/test/test.arff").numInstances());
    }
}
