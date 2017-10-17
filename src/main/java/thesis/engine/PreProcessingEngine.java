package thesis.engine;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class PreProcessingEngine {

    private Instances dataSet;

    public PreProcessingEngine(Instances dataSet) {
        this.dataSet = dataSet;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public void preProcess(String[] unneededAttributeNames, String[] replaceableAttributeNames,
                           double minNormalizeValue, double maxNormalizeValue)  {
        computeNewClassAttribute("satisfaction");
        addClassAttributeToDataSet(dataSet.attribute(dataSet.numAttributes()));
        filterUnneededAttributes(unneededAttributeNames);
        replaceMissingAttributeValues(parseAttributeNames(replaceableAttributeNames));
        normalizeAttributeValues(minNormalizeValue, maxNormalizeValue);
        //normalizeClassAttributeValues();
        balanceDataSet();
    }

    public void computeNewClassAttribute(String name) {
        List<String> classAttributeValues = new ArrayList<>();
        classAttributeValues.add("unsatisfied");
        classAttributeValues.add("neutral");
        classAttributeValues.add("satisfied");
        dataSet.insertAttributeAt(new Attribute(name, classAttributeValues), dataSet.numAttributes());

        Collections.list(dataSet.enumerateInstances())
                .forEach(instance -> {
                    double rating = instance.value(getAttributeIndex("rating"));
                    double recommendationScore = instance.value(getAttributeIndex("recommendation_score"));
                    double overallSatisfaction = (rating + recommendationScore) / 2;
                    if (overallSatisfaction < 5.0) {
                        instance.setValue(dataSet.numAttributes() - 1, "unsatisfied");
                    } else if (overallSatisfaction >= 6.5) {
                        instance.setValue(dataSet.numAttributes() - 1, "satisfied");
                    } else {
                        instance.setValue(dataSet.numAttributes() - 1, "neutral");
                    }
                });
    }

    public void addClassAttributeToDataSet(Attribute classAttribute) {
        if (dataSet.classIndex() == -1) {
            dataSet.setClass(classAttribute);
        }
    }

    private Set<Attribute> parseAttributeNames(String[] attributeNames) {
        return Stream.of(attributeNames)
                .map(attributeName ->
                        Collections.list(dataSet.enumerateAttributes())
                                .stream()
                                .filter(attr -> attr.name().equalsIgnoreCase(attributeName))
                                .findFirst()
                                .get())
                .collect(Collectors.toSet());
    }

    void filterUnneededAttributes(String[] attributeNames) {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(getAttributeIndices(attributeNames));

        try {
            removeFilter.setInputFormat(dataSet);
            this.dataSet = Filter.useFilter(dataSet, removeFilter);
        } catch (Exception e) {
            throw new RuntimeException("An error occurred when trying to remove attributes", e);
        }
    }

    private int[] getAttributeIndices(String[] attributeNames) {
        return Stream
                .of(attributeNames)
                .map(dataSet::attribute)
                .mapToInt(Attribute::index)
                .map(index -> index--).toArray();
    }

    private int getAttributeIndex(String attributeName) {
        return Collections.list(dataSet.enumerateAttributes())
                .stream()
                .filter(attribute -> attribute.name().equalsIgnoreCase(attributeName))
                .mapToInt(Attribute::index)
                .findFirst()
                .orElseThrow(() -> new RuntimeException("An error occurred when trying to get attribute index for attribute " + attributeName));
    }

    void replaceMissingAttributeValues(Set<Attribute> replaceableAttributes) {
        Enumeration<Instance> instanceEnumeration = dataSet.enumerateInstances();

        while (instanceEnumeration.hasMoreElements()) {
            Instance instance = instanceEnumeration.nextElement();

            for (int i = 0; i < instance.numAttributes(); i++) {
                Attribute currentAttribute = instance.attribute(i);
                if (instance.isMissing(i) && replaceableAttributes.contains(currentAttribute)) {
                    instance.setValue(currentAttribute, getReplacementValue(currentAttribute));
                }
            }
        }
    }

    void normalizeAttributeValues(double min, double max) {
        Collections.list(dataSet.enumerateInstances())
                .forEach(instance -> Collections.list(instance.enumerateAttributes())
                        .stream()
                        .filter(attribute -> attribute.index() != dataSet.classIndex())
                        .forEach(attribute -> {
                    double minOld = getExtremeValueOfAttribute(attribute, this::getMinValueOfStream);
                    double maxOld = getExtremeValueOfAttribute(attribute, this::getMaxValueOfStream);
                    double normalizedAttributeValue = getNormalizedAttributeValue(
                            instance.value(attribute),
                            minOld, maxOld,
                            min, max);
                    instance.setValue(attribute, normalizedAttributeValue);
                }));
    }

    void normalizeClassAttributeValues() {
        Collections.list(dataSet.enumerateInstances())
                .forEach(instance -> instance.setClassValue(instance.classValue() < 4 ? 0 : 1));

        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(dataSet.classIndex() + 1)); // 1 based index

        try {
            numericToNominal.setInputFormat(dataSet);
            this.dataSet = Filter.useFilter(dataSet, numericToNominal);
        } catch (Exception e) {
            throw new RuntimeException("Could not make class attribute " + dataSet.classAttribute().name() + " nominal.", e);
        }
    }

    void balanceDataSet() {
        int numInstancesDissatisfied = (int) filterInstancesByClassValue(0).count();
        List<Instance> instancesSatisfied = filterInstancesByClassValue(1).collect(Collectors.toList());
        Collections.shuffle(instancesSatisfied);
        instancesSatisfied = instancesSatisfied.subList(0, numInstancesDissatisfied);
        this.dataSet.removeIf(instance -> instance.classValue() == 1);
        this.dataSet.addAll(instancesSatisfied);
    }

    private Stream<Instance> filterInstancesByClassValue(double classValue) {
        return Collections
                .list(dataSet.enumerateInstances())
                .stream()
                .filter(instance -> instance.classValue() == classValue);
    }

    private double getExtremeValueOfAttribute(Attribute attribute, ToDoubleFunction<DoubleStream> getExtremeValueOfStream) {
        return getExtremeValueOfStream.applyAsDouble(Collections.list(dataSet.enumerateInstances())
                .stream()
                .filter(instance -> !instance.isMissing(attribute))
                .mapToDouble(instance -> instance.value(attribute)));
    }

    private double getMaxValueOfStream(DoubleStream stream) {
        return stream.max().orElseThrow(() -> new RuntimeException("No max value could be found in stream."));
    }

    private double getMinValueOfStream(DoubleStream stream) {
        return stream.min().orElseThrow(() -> new RuntimeException("No min value could be found in stream."));
    }

    private double getNormalizedAttributeValue(double value, double minOld, double maxOld, double minNew, double maxNew) {
        return ((value - minOld) / (maxOld - minOld)) * (maxNew - minNew) + minNew;
    }

    private double getReplacementValue(Attribute attribute) {
        return Collections.list(dataSet.enumerateInstances())
                .stream()
                .filter(instance -> !instance.isMissing(attribute))
                .collect(Collectors.averagingDouble(x -> x.value(attribute)));
    }

}
