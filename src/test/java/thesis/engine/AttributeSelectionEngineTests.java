package thesis.engine;

import org.junit.Assert;
import org.junit.Test;
import thesis.engine.selection.AttributeSelectionStrategy;
import thesis.engine.selection.CorrelationBasedSelectionStrategy;

import java.util.Collections;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class AttributeSelectionEngineTests {
    private static final String FILENAME = "data/test/test.arff";
    private final AttributeSelectionEngine attributeSelectionEngine;

    public AttributeSelectionEngineTests() {
        this.attributeSelectionEngine = new AttributeSelectionEngine(FileProcessingEngine.readDataSetFromFile(FILENAME));
    }

    @Test
    public void testFilterUnsuitableAttributes() {
        final int numAttributesBeforeFiltering = attributeSelectionEngine.getDataSet().numAttributes();
        final AttributeSelectionStrategy strategy = mock(CorrelationBasedSelectionStrategy.class);
        when(strategy.getAttributesToRemove()).thenReturn(new String[] {"gsm_rssi", "pos_uncertainty"});

        attributeSelectionEngine.filterUnsuitableAttributes(
                strategy);

        Assert.assertEquals(numAttributesBeforeFiltering - 2,
                attributeSelectionEngine.getDataSet().numAttributes());
        Assert.assertFalse(
                Collections.list(attributeSelectionEngine.getDataSet().enumerateAttributes())
                        .stream()
                        .anyMatch(attribute -> attribute.name().equalsIgnoreCase("gsm_rssi") ||
                                  attribute.name().equalsIgnoreCase("pos_uncertainty")));
    }
}
