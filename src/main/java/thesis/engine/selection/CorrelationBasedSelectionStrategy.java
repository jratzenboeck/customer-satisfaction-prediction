package thesis.engine.selection;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.CombinatoricsUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class CorrelationBasedSelectionStrategy extends AttributeSelectionStrategy {

    class Correlation {
        private final Set<Attribute> attributeSet;
        private final double correlationCoefficient;

        Correlation(Set<Attribute> attributeSet, double correlationCoefficient) {
            this.attributeSet = attributeSet;
            this.correlationCoefficient = correlationCoefficient;
        }

        double getCorrelationCoefficient() {
            return correlationCoefficient;
        }
    }

    CorrelationBasedSelectionStrategy(Instances dataSet, int minSetSize, int maxSetSize, int maxIterations) {
        super(dataSet, minSetSize, maxSetSize, maxIterations);
    }

    @Override
    Set<Attribute> getBestAttributeSet(List<Set<Attribute>> attributeSubSets) {
        List<Correlation> correlationsForSubsets = attributeSubSets
                .stream()
                .map(this::calculateCorrelation)
                .collect(Collectors.toList());
        correlationsForSubsets.sort((corr1, corr2) ->
                (int) Math.round(Math.abs(corr1.correlationCoefficient) - Math.abs(corr2.correlationCoefficient)));
        Collections.reverse(correlationsForSubsets);

        return correlationsForSubsets.get(0).attributeSet;
    }

    Correlation calculateCorrelation(Set<Attribute> attributeSubSet) {
        double interCorrelation = calculateInterCorrelation(attributeSubSet, getClassAttributeValues());
        double intraCorrelation = calculateIntraCorrelation(attributeSubSet);

        return new Correlation(attributeSubSet, interCorrelation / intraCorrelation);
    }

    double calculateInterCorrelation(Set<Attribute> attributeSubSet, double[] classAttributeValues) {
        return attributeSubSet
                .stream()
                .map(attribute -> Collections.list(getDataSet().enumerateInstances())
                        .stream()
                        .mapToDouble(instance -> instance.value(attribute))
                        .toArray())
                .collect(Collectors.averagingDouble(attributeValues ->
                        new PearsonsCorrelation().correlation(attributeValues, classAttributeValues)));
    }

    double calculateIntraCorrelation(Set<Attribute> attributeSubSet) {
        List<Attribute> attributes = new ArrayList<>();
        attributes.addAll(attributeSubSet);

        double sumCorrelationCoefficients = 0;

        for (int i = 0; i < attributes.size() - 1; i++) {
            double[] valuesOfFirstAttribute = getDataSet().attributeToDoubleArray(attributes.get(i).index());
            for (int j = i + 1; j < attributes.size(); j++) {
                double[] valuesOfSecondAttribute = getDataSet().attributeToDoubleArray(attributes.get(j).index());
                sumCorrelationCoefficients += new PearsonsCorrelation().correlation(valuesOfFirstAttribute, valuesOfSecondAttribute);
            }
        }
        return sumCorrelationCoefficients / CombinatoricsUtils.factorialDouble(attributeSubSet.size() - 1);
    }

    private double[] getClassAttributeValues() {
        return Collections.list(getDataSet().enumerateInstances())
                .stream()
                .mapToDouble(Instance::classValue)
                .toArray();
    }

}
