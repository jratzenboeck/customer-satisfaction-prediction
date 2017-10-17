package thesis.engine.selection;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public abstract class AttributeSelectionStrategy {

    private final int minSetSize, maxSetSize, maxIterations;
    private final Instances dataSet;

    AttributeSelectionStrategy(Instances dataSet, int minSetSize, int maxSetSize, int maxIterations) {
        this.dataSet = dataSet;
        this.minSetSize = minSetSize;
        this.maxSetSize = maxSetSize;
        this.maxIterations = maxIterations;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public String[] getAttributesToRemove() {
        List<Set<Attribute>> attributeSubSets = generateAttributeSubsets();
        Set<Attribute> bestAttributeSet = getBestAttributeSet(attributeSubSets);
        Set<Attribute> attributesToRemove = findAttributesToRemove(bestAttributeSet);

        return attributesToRemove.stream().map(Attribute::name).toArray(String[]::new);
    }

    List<Set<Attribute>> generateAttributeSubsets() {
        List<Set<Attribute>> attributeSubSets = new ArrayList<>();
        ThreadLocalRandom random = ThreadLocalRandom.current();

        for (int i = 0; i < maxIterations; i++) {
            Set<Attribute> currentAttributeSet = new HashSet<>();

            for (int j = 0; j <= random.nextInt(minSetSize, maxSetSize); j++) {
                int randomAttributeIndex = random.nextInt(0, dataSet.numAttributes());
                currentAttributeSet.add(dataSet.attribute(randomAttributeIndex));
            }
            attributeSubSets.add(currentAttributeSet);
        }
        return attributeSubSets;
    }

    Set<Attribute> findAttributesToRemove(Set<Attribute> bestAttributeSet) {
        return Collections.list(dataSet.enumerateAttributes())
                .stream()
                .filter(attribute -> !bestAttributeSet.contains(attribute))
                .collect(Collectors.toSet());
    }

    abstract Set<Attribute> getBestAttributeSet(List<Set<Attribute>> attributeSubSets);
}
