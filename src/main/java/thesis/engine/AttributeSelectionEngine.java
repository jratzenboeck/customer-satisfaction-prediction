package thesis.engine;

import thesis.engine.selection.AttributeSelectionStrategy;
import weka.core.Instances;

public class AttributeSelectionEngine {

    private Instances dataSet;

    public AttributeSelectionEngine(Instances dataSet) {
        this.dataSet = dataSet;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public void filterUnsuitableAttributes(AttributeSelectionStrategy strategy) {
        String[] attributesToRemove = strategy.getAttributesToRemove();

        PreProcessingEngine preProcessingEngine = new PreProcessingEngine(dataSet);
        preProcessingEngine.filterUnneededAttributes(attributesToRemove);
        this.dataSet = preProcessingEngine.getDataSet();
    }
}
