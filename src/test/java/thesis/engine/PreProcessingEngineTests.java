package thesis.engine;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;

public class PreProcessingEngineTests {

    private static final String FILENAME = "data/test/test.arff";
    private PreProcessingEngine preProcessingEngine;

    @Before
    public void setUp() {
        preProcessingEngine = new PreProcessingEngine(FileProcessingEngine.readDataSetFromFile(FILENAME));
    }

    private Instances getDataSet() {
        return preProcessingEngine.getDataSet();
    }

    @Test
    public void testAddClassAttributeToDataSet() {
        preProcessingEngine.addClassAttributeToDataSet(getDataSet().attribute("rating"));
        Assert.assertEquals(1, getDataSet().classIndex());
    }

    @Test
    public void testFilterUnneededAttributes() {
        int numAttributesBeforeFiltering = getDataSet().numAttributes();
        String[] attributesToFilter =  new String[] {"submit_date", "created_at"};
        int numAttributesAfterFiltering = numAttributesBeforeFiltering - attributesToFilter.length;

        preProcessingEngine.filterUnneededAttributes(attributesToFilter);
        Assert.assertEquals(numAttributesAfterFiltering, getDataSet().numAttributes());
        Assert.assertNull(getDataSet().attribute("submit_date"));
        Assert.assertNull(getDataSet().attribute("created_at"));
    }

    @Test
    public void testReplaceMissingValues() {
        Set<Attribute> replaceableAttributes = new HashSet<>();
        Attribute attribute = getDataSet().attribute("gsm_rssi");
        replaceableAttributes.add(attribute);

        preProcessingEngine.replaceMissingAttributeValues(replaceableAttributes);
        boolean containsMissingValues = Collections
                .list(getDataSet().enumerateInstances())
                .stream()
                .anyMatch(instance -> instance.isMissing(attribute));

        Assert.assertFalse(containsMissingValues);
    }

    @Test
    public void testNormalizeAttributeValues() {
        final double minValue = 0.0;
        final double maxValue = 1.0;
        preProcessingEngine.addClassAttributeToDataSet(getDataSet().attribute("rating"));

        preProcessingEngine.normalizeAttributeValues(minValue, maxValue);

        boolean isAttrValueOutsideRange = hasDataSetAttrValuesOutsideRange(
                getDataSet(),
                attribute -> attribute.index() != getDataSet().classIndex(),
                value -> value < minValue || value > maxValue);

        Assert.assertFalse(isAttrValueOutsideRange);
    }

    @Test
    public void testNormalizeClassAttributeValues() {
        preProcessingEngine.addClassAttributeToDataSet(getDataSet().attribute("rating"));
        preProcessingEngine.normalizeClassAttributeValues();

        boolean isClassAttrValueOutsideRange = hasDataSetAttrValuesOutsideRange(
                getDataSet(),
                attribute -> attribute.index() == getDataSet().classIndex(),
                value -> value == 0 || value == 1);

        Assert.assertTrue(getDataSet().classAttribute().isNominal());
        Assert.assertFalse(isClassAttrValueOutsideRange);
    }

    private boolean hasDataSetAttrValuesOutsideRange(Instances dataSet, Predicate<Attribute> filterCondition, Predicate<Double> valueCondition) {
        return Collections.list(dataSet.enumerateInstances())
                .stream()
                .anyMatch(instance ->
                        hasInstanceAttrValueOutsideRange(instance,
                                filterCondition,
                                valueCondition));

    }
    private boolean hasInstanceAttrValueOutsideRange(Instance instance, Predicate<Attribute> filterCondition, Predicate<Double> valueCondition) {
        return Collections.list(instance.enumerateAttributes())
                .stream()
                .filter(filterCondition)
                .map(instance::value)
                .anyMatch(valueCondition);
    }
}
