# use bar chart to show the distribution of misclassified labels with different methods.
# method 1's misclassified labels results [38: 449, 39: 20, 40: 20, 41: 20, 42: 20, 43: 149, 44: 222]
# method 2's misclassified labels results [38: 645, 39: 20, 40: 20, 41: 20, 42: 20, 43: 149, 44: 222]
# use method1's order when plotting the bar chart.
import matplotlib.pyplot as plt

method1_data = {38.0: 449, 3.0: 180, 44.0: 178, 5.0: 164, 43.0: 140, 2.0: 135, 0.0: 107, 17.0: 79, 36.0: 69, 12.0: 64, 13.0: 60, 1.0: 56, 10.0: 49, 31.0: 36, 14.0: 35, 7.0: 31, 26.0: 29, 11.0: 19, 29.0: 19, 21.0: 18, 37.0: 17, 35.0: 17, 25.0: 17, 4.0: 17, 34.0: 16, 9.0: 16, 39.0: 15, 15.0: 15, 33.0: 14, 22.0: 14, 16.0: 14, 30.0: 13, 28.0: 12, 24.0: 11, 18.0: 11, 41.0: 10, 27.0: 8, 23.0: 6, 40.0: 5, 19.0: 4, 6.0: 3, 32.0: 3, 8.0: 2, 20.0: 1}
method2_data = {38.0: 234, 43.0: 185, 5.0: 169, 3.0: 147, 44.0: 133, 2.0: 132, 0.0: 129, 17.0: 129, 14.0: 93, 12.0: 80, 1.0: 77, 36.0: 75, 10.0: 61, 35.0: 51, 16.0: 48, 26.0: 42, 7.0: 40, 22.0: 36, 37.0: 35, 13.0: 30, 31.0: 28, 28.0: 27, 9.0: 24, 25.0: 22, 15.0: 21, 4.0: 21, 34.0: 19, 21.0: 14, 33.0: 14, 29.0: 13, 23.0: 13, 41.0: 13, 30.0: 11, 39.0: 11, 24.0: 11, 18.0: 10, 11.0: 8, 6.0: 7, 32.0: 7, 40.0: 6, 27.0: 6, 19.0: 5, 8.0: 2, 20.0: 1}

#sort the data by the key
method1_data = dict(sorted(method1_data.items()))

# Sort method2_data according to method1_data's order
method2_data_sorted = {k: method2_data[k] for k in method1_data.keys()}

# Data for plotting
labels = list(method1_data.keys())
method1_values = list(method1_data.values())
method2_values = [method2_data_sorted[k] for k in labels]

# Set up the bar chart
x = range(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x, method1_values, width, label='without weights')
bars2 = ax.bar([p + width for p in x], method2_values, width, label='with weights')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Labels')
ax.set_ylabel('Misclassifications')
ax.set_title('Misclassified labels by method')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

# Rotate the tick labels for better readability
plt.xticks(rotation=90)

plt.show()