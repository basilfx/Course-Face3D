"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

from matplotlib import pyplot
from scipy import signal

import numpy
import scipy
import math
import os

def max_xy(input, offset_x=0, offset_y=0):
    index = numpy.nanargmax(input)
    y, x = numpy.unravel_index(index, input.shape)
    return x + offset_x, y + offset_y

def min_xy(input, offset_x=0, offset_y=0):
    index = numpy.nanargmin(input)
    y, x = numpy.unravel_index(index, input.shape)
    return x + offset_x, y + offset_y

def find_peak_start(input, treshold=0.001, peak_length=15):
    input = numpy.abs(numpy.gradient(signal.bspline(input, 25)))

    # Control parameters
    input_treshold = numpy.nanmax(input) * treshold
    input_length = input.shape[0]
    recording = False
    start_x = 0
    stop_x = 0

    # Walk from start to end. When the current value exceeds threshold,
    # start recording.
    for i in range(0, input_length):
        if recording:
            if input[i] > treshold:
                stop_x = i

                if (stop_x - start_x) > peak_length:
                    return start_x
            else:
                recording = False
        else:
            if input[i] > treshold:
                start_x = i
                recording = True

    # Nothing found
    return 0

def find_peak_stop(input, *args):
    # Apply the start search but reverse array
    x = find_peak_start(input[::1], *args)

    # Reverse result
    return input.shape[0] - x

def face(index=0, fit=True, N=67, K=12, interpolation=True):
    from absfile import AbsFile
    from face import Face
    import algorithms

    file = ['test_16/04371d164.abs', 'test_same/04203d350.abs', 'test_same/04203d352.abs', 'test_same/04203d354.abs', 'test_8/04316d216.abs', 'test_4/04374d193.abs'][index]
    face = Face(AbsFile(file))
    
    if interpolation:
        algorithms.repair(face)

    algorithms.smooth(face)
    algorithms.crop(face)
    algorithms.zoom(face)
    algorithms.key_points(face)
    algorithms.rotate(face)

    if fit:
        algorithms.key_points(face)
        algorithms.fit(face)

    # Extract features
    algorithms.features_histogram(face, N, K)

    return face

def evaluate_interpolation(output_file='interpolate.pdf'):
    f = face(5, interpolation=False)
    g = face(5)
    figure = pyplot.figure()

    subplot = pyplot.subplot(1, 2, 1)
    subplot.imshow(f.Z)

    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 2, 2)
    subplot.imshow(g.Z)

    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

def evaluate_features(output_file='features.pdf'):
    import algorithms
    blue = (0.0, 0.0, 1.0, 1.0)
    g = face(1, True, 18, 12)
    h = face(2, True, 18, 12)
    f = face(3, True, 18, 12)
    k = face(0, True, 18, 12)

    figure = pyplot.figure()

    subplot = pyplot.subplot(1, 5, 1)

    subplot.imshow(g.Z)

    for xx in range(1, 8):
        v = int((xx) * (g.height / 8))
        subplot.axhline(v, color=blue)

    subplot.set_xlabel('Different regions (N=8)')
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 5, 2)
    subplot.imshow(g.features[1].reshape(g.features[2]))
    subplot.set_xlabel('Person 1a')
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 5, 3)
    subplot.imshow(h.features[1].reshape(h.features[2]))
    subplot.set_xlabel('Person 1b')
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 5, 4)
    subplot.imshow(f.features[1].reshape(f.features[2]))
    subplot.set_xlabel('Person 1c')
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 5, 5)
    subplot.imshow(k.features[1].reshape(k.features[2]))
    subplot.set_xlabel('Person 2a') 
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)   

    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

def evaluate_feature_extraction(output_file='face.pdf', output2_file='surfature.pdf', output3_file='mouth.pdf'):
    import algorithms
    f = face(4, fit=False)
    g = face(4)
    h = face(1)
    grey = (0.7, 0.7, 0.7, 0.7)
    K, H, Pmax, Pmin = algorithms.surfature(f)

    ########

    figure = pyplot.figure()

    subplot = pyplot.subplot(1, 1, 1)
    subplot.imshow(h.abs_file.data['Z'])

    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    figure.savefig("raw.pdf", format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

    ########

    figure = pyplot.figure()

    subplot = pyplot.subplot(1, 1, 1)
    subplot.imshow(f.Z)

    for name, point in f.key_points.iteritems():
        x, y = point
        subplot.plot(x, y, 'x', color='k') 

    nose_x, nose_y = f.key_points['nose']
    lip_x, lip_y = f.key_points['lip']
    chin_x, chin_y = f.key_points['chin']

    nose_left_x, nose_left_y = f.key_points['nose_left']
    nose_right_x, nose_right_y = f.key_points['nose_right']
    lip_left_x, lip_left_y = f.key_points['lip_left']
    lip_right_x, lip_right_y = f.key_points['lip_right']

    border_lip_left_x, border_lip_left_y = f.key_points['border_lip_left']
    border_lip_right_x, border_lip_right_y = f.key_points['border_lip_right']

    pyplot.plot([nose_left_x, nose_right_x], [nose_left_y, nose_right_y], color="K")
    pyplot.plot([lip_left_x, lip_right_x], [lip_left_y, lip_right_y], color="K")

    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

    ########

    figure = pyplot.figure()

    subplot = pyplot.subplot(1, 1, 1)
    subplot.imshow(g.Z)

    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    figure.savefig("face2.pdf", format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

    ########

    figure = pyplot.figure(figsize=(10, 5))

    subplot = pyplot.subplot(1, 1, 1)
    #subplot.plot(K[:, nose_x])
    a, = subplot.plot(H[:, nose_x] * 2)
    #subplot.plot(Pmin[:, nose_x])
    b, = subplot.plot(Pmax[:, nose_x])
    c, = subplot.plot(f.Z[:, nose_x] / 50)

    subplot.set_xlabel('Vertical Face Position')
    subplot.set_ylabel('Value')
    subplot.axvline(nose_y, color=grey)
    subplot.axvline(lip_y, color=grey)
    subplot.axvline(chin_y, color=grey)
    subplot.legend([a, b, c], ["Mean", "Pmax", "Original"])

    figure.show()
    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

    ##########
    figure = pyplot.figure(figsize=(10, 5))

    subplot = pyplot.subplot(1, 2, 1)

    #a, = subplot.plot(Pmax[(lip_y - 5):(lip_y + 5), :])
    a, = subplot.plot(H[lip_y, :])
    b, = subplot.plot(Pmax[lip_y, :] * 2)
    c, = subplot.plot(f.Z[lip_y, :] / 20)

    subplot.set_xlabel('Horizontal Face Position')
    subplot.set_ylabel('Value')
    subplot.axvline(lip_right_x, color=grey)
    subplot.axvline(lip_left_x, color=grey)
    subplot.axvline(border_lip_left_x, color=grey)
    subplot.axvline(border_lip_right_x, color=grey)
    subplot.legend([a, b, c], ["Mean", "Pmax", "Original"])

    subplot = pyplot.subplot(1, 2, 2)

    a, = subplot.plot(numpy.nansum(H[(lip_y - 5):(lip_y + 5), :], axis=0))
    b, = subplot.plot(numpy.nansum(Pmax[(lip_y - 5):(lip_y + 5), :], axis=0) * 2)
    c, = subplot.plot(numpy.nansum(f.Z[(lip_y - 5):(lip_y + 5), :], axis=0) / 100)

    subplot.set_xlabel('Horizontal Face Position (summed)')
    subplot.set_ylabel('Value')
    subplot.axvline(lip_right_x, color=grey)
    subplot.axvline(lip_left_x, color=grey)
    subplot.axvline(border_lip_left_x, color=grey)
    subplot.axvline(border_lip_right_x, color=grey)
    subplot.legend([a, b, c], ["Mean", "Pmax", "Original"])

    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")


def evaluate_rotate(rotations=[-5.0, -2.5, -1.0, 1, 2.5, 5.0], index=4, output_file='rotations.pdf'):
    from scipy.ndimage import interpolation
    import algorithms

    original = face(index)
    other = face(1)
    faces = []

    for rotation in rotations:
        f = face(index)

        f.abs_file.data['X'] = interpolation.rotate(f.abs_file.data['X'], rotation, mode='nearest', prefilter=False, reshape=False)
        f.abs_file.data['Y'] = interpolation.rotate(f.abs_file.data['Y'], rotation, mode='nearest', prefilter=False, reshape=False)
        f.abs_file.data['Z'] = interpolation.rotate(f.abs_file.data['Z'], rotation, mode='nearest', prefilter=False, reshape=False)

        algorithms.features_histogram(f)
        faces.append(f)

    pyplot.figure()

    subplot = pyplot.subplot(1, 2+len(rotations), 1)

    subplot.imshow(original.Z)
    subplot.title.set_text("Original")
    subplot.title.set_fontsize(10)
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    for rotation, f, i in zip(rotations, faces, range(len(rotations))):
        subplot = pyplot.subplot(1, 2+len(rotations), 2 + i)
        subplot.imshow(f.Z)
        subplot.title.set_text("%.1f deg" % rotation)
        subplot.title.set_fontsize(10)
        subplot.xaxis.set_visible(False)
        subplot.yaxis.set_visible(False)

    subplot = pyplot.subplot(1, 2+len(rotations), len(rotations) + 2)
    subplot.imshow(other.Z)
    subplot.title.set_text("Other")
    subplot.title.set_fontsize(10)
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)

    pyplot.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

    return algorithms.similarity_matrix([original] + faces + [other], methods=[algorithms.distance_histogram_euclidean, algorithms.distance_histogram_city_block, algorithms.distance_histogram_correlation], normalizers=[False, False, False])

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = numpy.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord

    return data.sum()

def intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""

    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = numpy.array([[u20, u11], [u11, u02]])

    return x_bar, y_bar, cov

def generate_base_map(faces, func, output_file):
    # Make sure there is data
    if not faces or len(faces) == 0:
        print "Nothing to do"
        return

    # Calculate rows and columns
    columns = math.ceil(math.sqrt(len(faces)))
    rows = ((len(faces) - 1) / columns) + 1;
    figure = pyplot.figure()
    figure.subplots_adjust(top=0.85)
    index = 1

    for (file, person_id, face) in faces:
        # Advance to next plot
        subplot = figure.add_subplot(columns, rows, index, xticks=[], yticks=[])
        index = index + 1

        # Plot face
        func(subplot, file, person_id, face, index)
        subplot.title.set_text(person_id)
        subplot.title.set_fontsize(10)
        subplot.xaxis.set_visible(False)
        subplot.yaxis.set_visible(False)

    # Save figure
    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

def generate_feature_map(faces, output_file):
    def _func(subplot, file, person_id, face, index):
        subplot.imshow(face.features[1].reshape(face.features[2]))

    generate_base_map(faces, _func, output_file)

def generate_depth_map(faces, output_file, key_points=False):
    def _func(subplot, file, person_id, face, index):
        subplot.imshow(face.Z)

        if key_points == True:
            for name, point in face.key_points.iteritems():
                x, y = point
                subplot.plot(x, y, 'x', color='k')

    generate_base_map(faces, _func, output_file)

def generate_similarity_matrix(matrix, faces, output_file):
    methods, rows, cols = matrix.shape
    output = []

    # Iterate each method
    for i in range(methods):
        if cols > 0:
            table = []
            table.append("<h1>Method %d</h1><table><tr><td></td>%s</tr>" % (i, "\n".join([ "<td>%s</td>" % faces[j][1] for j in range(cols) ])))

            for j in range(rows):
                table.append("<tr><td>%s</td>%s</tr>" % (faces[j][1], "\n".join([ "<td>%s</td>" % (("%.2f" % matrix[(i, j, k)]) if not numpy.isnan(matrix[(i, j, k)]) else "&mdash;") for k in range(cols) ])))

            table.append("</table>")
            output.append("\n".join(table))

    # Write table to file
    with open(output_file, "w") as f:
        f.write("""
            <html>
                <head>
                    <title>Similarity Matrix</title>
                </head>
                <body>
                    %s
                </body>
            </html>
        """ % "\n".join(output))

def generate_roc_curve(rocs, output_file):
    grey = (0.7, 0.7, 0.7, 0.7)
    figure = pyplot.figure()
    titles = ["Euclidean", "City Block", "Correlation"]
    legends = []
    plots = []
    index = 0

    # Draw ROC line
    subplot = pyplot.subplot(1, 1, 1)
    subplot.plot([0, 1], [1, 0], color=grey)

    # Plot each line
    for roc, eer in rocs:
        plots.extend(subplot.plot(roc[0], roc[1]))
        subplot.plot(eer, 1 - eer, 'x', color='r')

        # Include EER in legend
        legends.append("%s (EER=%.2f%%)" % (titles[index], eer * 100))
        index = index + 1

    # Axis and legend
    subplot.set_xlabel('False positives rate')
    subplot.set_ylabel('True positives rate')
    subplot.legend(plots, legends, loc=4)

    # Save figure
    figure.savefig(output_file, format='pdf', dpi=600, orientation='landscape', bbox_inches="tight")

class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self): 
        return self.length

    def __iter__(self):
        return self.gen

