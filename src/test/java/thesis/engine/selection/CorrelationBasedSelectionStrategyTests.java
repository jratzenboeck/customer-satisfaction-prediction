package thesis.engine.selection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import thesis.engine.FileProcessingEngine;
import thesis.engine.PreProcessingEngine;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CorrelationBasedSelectionStrategyTests {

    private static final String FILENAME = "data/test/test.arff";
    private final Instances dataSet;
    private final CorrelationBasedSelectionStrategy strategy;
    private Set<Attribute> featureSet;

    public CorrelationBasedSelectionStrategyTests() {
        dataSet = FileProcessingEngine.readDataSetFromFile(FILENAME);
        strategy = new CorrelationBasedSelectionStrategy(dataSet, 1, 1, 1);
        new PreProcessingEngine(dataSet).addClassAttributeToDataSet(dataSet.attribute("rating"));
    }

    @Before
    public void setUp() {
        featureSet = new HashSet<>();
    }

    @Test
    public void testCalculateInterCorrelation() {
        featureSet.add(dataSet.attribute("gsm_rssi"));

        double correlation = strategy.calculateInterCorrelation(featureSet, new double[] {4, 5, 1});

        Assert.assertEquals(0.2064638, correlation, 0.0001);
    }

    @Test
    public void testCalculateIntraCorrelation() {
        createTestFeatureSet();

        double correlation = strategy.calculateIntraCorrelation(featureSet);

        Assert.assertEquals(-0.961014, correlation, 0.0001);
    }

    @Test
    public void testCalculateCorrelation() {
        createTestFeatureSet();

        CorrelationBasedSelectionStrategy.Correlation correlation = strategy.calculateCorrelation(featureSet);

        Assert.assertEquals(-0.1449466, correlation.getCorrelationCoefficient(), 0.0001);
    }

    @Test
    public void testGetBestAttributeSet() {
        createTestFeatureSet();

        Set<Attribute> featureSet2 = new HashSet<>();
        featureSet2.add(dataSet.attribute("cmd_success_rate"));
        featureSet2.add(dataSet.attribute("cmd_delay_to_pos_new"));

        List<Set<Attribute>> attributeSets = new ArrayList<>();
        attributeSets.add(featureSet);
        attributeSets.add(featureSet2);

        Set<Attribute> bestAttributeSet = strategy.getBestAttributeSet(attributeSets);

        Assert.assertArrayEquals(featureSet2.stream().map(Attribute::name).toArray(),
                bestAttributeSet.stream().map(Attribute::name).toArray());
    }

    private void createTestFeatureSet() {
        featureSet.add(dataSet.attribute("gsm_rssi"));
        featureSet.add(dataSet.attribute("pos_uncertainty"));
    }
}
