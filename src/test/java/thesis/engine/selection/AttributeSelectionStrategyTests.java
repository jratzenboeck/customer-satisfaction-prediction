package thesis.engine.selection;

import org.junit.Assert;
import org.junit.Test;
import thesis.engine.FileProcessingEngine;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class AttributeSelectionStrategyTests {

    private final AttributeSelectionStrategy strategy;
    private static final int ITERATIONS = 10;
    private static final int MIN_SET_SIZE = 2;
    private static final int MAX_SET_SIZE = 10;

    public AttributeSelectionStrategyTests() {
        Instances dataSet = FileProcessingEngine.readDataSetFromFile("data/test/test.arff");
        strategy = new CorrelationBasedSelectionStrategy(dataSet, MIN_SET_SIZE, MAX_SET_SIZE, ITERATIONS);
    }

    @Test
    public void testGenerateAttributeSubsets() {
        List<Set<Attribute>> attributeSubSets = strategy.generateAttributeSubsets();

        Assert.assertEquals(ITERATIONS, attributeSubSets.size());
        Assert.assertFalse(attributeSubSets
                .stream()
                .anyMatch(subSet -> subSet.size() < MIN_SET_SIZE || subSet.size() > MAX_SET_SIZE));
    }

    @Test
    public void testFindAttributesToRemove() {
        Set<Attribute> bestAttributeSet = new HashSet<>();
        bestAttributeSet.add(strategy.getDataSet().attribute("cmd_success_rate"));
        bestAttributeSet.add(strategy.getDataSet().attribute("cmd_delay_to_pos_new"));

        Set<Attribute> attributesToRemove = strategy.findAttributesToRemove(bestAttributeSet);

        Assert.assertEquals(strategy.getDataSet().numAttributes() - 2, attributesToRemove.size());
    }
}
