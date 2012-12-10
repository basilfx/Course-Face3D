"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

from face3d import AbsFile, Face, Database
from face3d import utils, algorithms

import argparse
import glob
import os
import sys
import numpy
import threading

parameters = {"N": 67, "K": 12}
parameter_types = {"N": int, "K": int}
database = None
debug = False

def main():
    global database, parameters
    success = False

    # Parse arguments
    arguments, parser = argument_parser()

    # Parse parameters
    try:
        for parameter in arguments.parameters.split(','):
            key, value = parameter.split('=', 1)
            parameters[key] = value
    except:
        print "Invalid parameters: %s" % arguments.parameters
        sys.exit(1)

    # Make sure parameters are of right type
    for key, value in parameters.iteritems():
        try:
            parameters[key] = parameter_types[key](value)
        except:
            print "Parameter '%s' of incorrect type" % key
            sys.exit(1)

    # Initialize a database
    database = Database(arguments.database)

    # Enrollment
    if arguments.enroll:
        success = True
        is_directory, path = arguments.enroll

        if is_directory:
            files = glob.glob("%s/*.abs" % path)
            thread_count = 16
            chunks = [ files[i::thread_count] for i in range(thread_count) ]
            threads = []

            # Process each thread
            for chunk in chunks:
                thread = threading.Thread(target=lambda x: [ enroll_face(c, arguments.person_id, arguments.auto_id) for c in x ], args=(chunk, ))
                thread.daemon = True
                threads.append(thread)
                thread.start()
            
            # Wait for the threads to finish
            for thread in threads:
                thread.join()

        else:
            enroll_face(path, arguments.person_id, arguments.auto_id)

    # Caches for 
    faces = False
    matrix = False
    rocs = False

    # Authenticating
    if arguments.authenticate:
        print "Authenticating face from '%s'" % arguments.authenticate
        success = True

        # Create face from file
        face = Face(AbsFile(arguments.authenticate))

        # Normalize it
        algorithms.process(face, parameters["N"], parameters["K"])

        # Get the other data
        if not faces: faces = list(database.iterator())
        matrix = algorithms.similarity_matrix([face] + [ face[2] for face in faces ], limit=1) # One line matrix

        # Evaluate result
        methods, _, _ = matrix.shape
        tresholds = [0.90, 0.90, 0.90]

        for i in range(methods):
            # Select indexes of candidates
            vector = numpy.array(matrix[i][0][1:])
            candidates, = numpy.where(vector >= tresholds[i])
            persons = {}

            # Verify candidates
            if len(candidates) == 0:
                print "Method %d does not yield any candidates!" % i
                continue

            # Print method
            print "Results for method %d:" % i

            # Print each candidate
            for candidate in candidates:
                if candidate == 0: 
                    continue

                filename, person_id, data = faces[candidate]

                # Add person to list of persons
                if person_id not in persons:
                    persons[person_id] = []

                persons[person_id].append(matrix[i][0][candidate + 1])

            # Print results
            for person, scores in persons.iteritems():
                print "Match with person %s with scores %s" % (person, [ "%.2f" % s for s in scores ])

    # Reevaluation
    if arguments.reevaluate:
        print "Reevaluate faces"
        success = True

        # Get data
        if not faces: faces = list(database.iterator())

        # Action
        [ algorithms.features_histogram(face[2], parameters["N"], parameters["K"]) for face in faces ]
        

    # Visualizing
    if arguments.depth_map:
        print "Generating depth map"
        success = True
        
        # Get data
        if not faces: faces = list(database.iterator())
        
        # Action
        utils.generate_depth_map(faces, arguments.depth_map, arguments.draw_key_points)
    
    if arguments.feature_map:
        print "Generating feature map"
        success = True

        # Get data
        if not faces: faces = list(database.iterator())

        # Action
        utils.generate_feature_map(faces, arguments.feature_map)

    if arguments.similarity_matrix:
        print "Generating similarity matrix"
        success = True

        # Get data
        if not faces: faces = list(database.iterator())
        if not matrix: matrix = algorithms.similarity_matrix([ face[2] for face in faces ])

        # Action
        utils.generate_similarity_matrix(matrix, faces, arguments.similarity_matrix)

    if arguments.roc_curve:
        print "Generating ROC curve"
        success = True

        # Get data
        if not faces: faces = list(database.iterator())
        if not matrix: matrix = algorithms.similarity_matrix([ face[2] for face in faces ])
        if not rocs: rocs = algorithms.calculate_roc_eer(matrix, [ face[1] for face in faces ])

        utils.generate_roc_curve(rocs, arguments.roc_curve)

    # Print help in case of no action
    if not success:
        parser.print_help()
        sys.exit(1)
    else:
        sys.exit(0)

def enroll_face(file, person_id=None, auto_id=False, force=False):
    filename = os.path.basename(file)
    
    # Check for duplicates
    if database.exists(file) and not force:
        print "File '%s' already enrolled" % filename
        return

    # Make sure we have an identifier
    if not person_id and not auto_id:
        print "Auto person identification disabled and no identification specified."
        return

    # File not yet enrolled
    print "Processing %s" % filename

    # Read data file
    absfile = AbsFile(file)

    # Derrive filename
    if auto_id:
        basename = os.path.basename(file)
        person_id = basename[:basename.index('d')]

    # Create Face object
    face = Face(absfile)

    # Apply algorithms to process raw data
    try:
        # Apply selected algorithms
        algorithms.process(face, parameters["N"], parameters["K"])

        # Compress data
        face.compress()
    except:
        print "File '%s' failed" % file
        
        # In debug mode, show exceptions
        if debug:
            raise
        else:
            return

    # Enroll to database
    database.save(file, person_id, face)

def argument_parser():
    # Helper for argparse to match file and/or directory
    def helper_file_or_directory(parser):
        def _inner(path):
            # First, resolve
            path = os.path.realpath(path)

            # Then check path
            if not os.path.exists(path):
                parser.error("The file or directory '%s' does not exist" % path)
            else:
                if os.path.isdir(path):
                    return (True, path)
                else:
                    return (False, open(path, "rb"))
        return _inner

    # Helper for argparse to match file
    def helper_file(parser):
        def _inner(file):
            # First, resolve
            path = os.path.realpath(file)

            # Then check path
            if not os.path.exists(file) or os.path.isdir(file):
                parser.error("The file '%s' does not exist" % path)

            # Done
            return file
        return _inner

    # Create parser
    parser = argparse.ArgumentParser(description='Enroll, match and visualize 3D faces')

    # Add the arguments
    parser.add_argument('-d', '--database', default='database.db', help='path to cache file')
    parser.add_argument('-p', '--parameters', default='K=12,N=67', action='store', help='algorithm parameters, comma seperated')

    group = parser.add_argument_group(title="Face management")
    group.add_argument('--authenticate', type=helper_file(parser), help='authenticate a face to enrolled faces')
    group.add_argument('--enroll', type=helper_file_or_directory(parser),  help='enroll face from file or directory')
    group.add_argument('--person-id', action='store', help='number or name identifing person')
    group.add_argument('--auto-id', action='store_true', help='derrive person identifier from filename')
    group.add_argument('--reevaluate', action='store_true', help='reevaluation enrolled faces, but do not save')

    group = parser.add_argument_group(title="Visualization")
    group.add_argument('--depth-map', action='store', help='generate a depth map of enrolled faces')
    group.add_argument('--feature-map', action='store', help='generate a feature map of enrolled faces')
    group.add_argument('--similarity-matrix', action='store', help='generate a similarity matrix of all implemented methods')
    group.add_argument('--roc-curve', action='store', help='generate a ROC curve of all implemented methods')
    group.add_argument('--draw-key-points', action='store_true', help='include key points on depth map')

    # Done
    return parser.parse_args(), parser

# Application main
if __name__ == '__main__':
    main()