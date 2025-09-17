from __future__ import division
from __future__ import absolute_import
import subprocess
import os
import sys
import numpy as np
from io import open
import argparse
import jinja2
import fiber_configs 


# Usage
# sample command: python Controller.py  -n 45 -e -45 -l 4 -p ./ProcessGround \
#                                       -s ../FiberSynthesizer/mesh_files/1_surf.txt
#                                       -t ../FiberSynthesizer/mesh_files/1_set.txt
#                                       -o ../FiberSynthesizer/mesh_files/1_node.txt
#                                       -m ../FiberSynthesizer/mesh_files/1_elem.txt

# full argnames example
# python Controller.py --geom 1 --endo 45 --epi -45 --layer 4 --path ./ProcessGround \
#                     --surf ../FiberSynthesizer/mesh_files/1_surf.txt
#                     --sets ../FiberSynthesizer/mesh_files/1_set.txt
#                     --node ../FiberSynthesizer/mesh_files/1_node.txt
#                     --elem ../FiberSynthesizer/mesh_files/1_elem.txt


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')


def process_dat_file(NElem, dat_file_address=None):
    
    DatFile = open(dat_file_address, u'r')
    DatFile.seek(0)
    DatLines = []
    print('@@ Reading' + dat_file_address + ' file...')
    sys.stdout.flush()
    for idx, line in enumerate(DatFile):
        DatLines.append(line.strip())
        if u'THE FOLLOWING TABLE IS PRINTED AT THE CENTROID' in line:
            TableFirstRow = idx + 6
            TableLastRow = TableFirstRow + NElem - 1
    DatFile.close()
    Dat = np.empty((NElem, 6))
    for idx, line in enumerate(DatLines[TableFirstRow:TableLastRow + 1]):
        Dat[idx] = np.fromstring(DatLines[TableFirstRow:TableLastRow + 1][idx], dtype=float ,sep=u' ')
    Temps_b = Dat[:, 0:2]
    return Temps_b


env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))

template = env.get_template('ht_analysis.inp')

INP_FILES = {
    u'HT_Analysis_BV_b': template.render(data=fiber_configs.data_b),
    u'HT_Analysis_BV_c': template.render(data=fiber_configs.data_c),
    u'HT_Analysis_BV_d': template.render(data=fiber_configs.data_d)
}


parser = argparse.ArgumentParser(description='Fiber Synthesizer Controller')


parser.add_argument('-g', '--geom', type=int, default=-1, help='Geometry Number')
parser.add_argument('-n', '--endo', type=float, required=True, help='LV Endo Angle')
parser.add_argument('-e', '--epi', type=float, required=True, help='LV Epi Angle')
parser.add_argument('-l', '--layer', type=int, default=4, help='Number of Layers')
parser.add_argument('-p', '--path', type=str, default=os.getcwd(), help='Path to Process Ground')
parser.add_argument('-s', '--surf', type=str, default='', help='Surface File')
parser.add_argument('-t', '--sets', type=str,  default='', help='Set File')
parser.add_argument('-o', '--node', type=str, default='', help='Node File')
parser.add_argument('-m', '--elem', type=str, default='', help='Element File')
# skip mesh manipulation argument default is False and is optional
parser.add_argument('-x', '--skip-mesh-manipulation', action='store_true', help='Skip Mesh Manipulation')
parser.add_argument('-y', '--skip-abaqus', action='store_true', help='Skip Abaqus Execution')
parser.add_argument('-z', '--skip-dat-analyse', action='store_true', help='Skip Data Processing')
# suffix
parser.add_argument('-u', '--suffix', type=int, default=0, help='Suffix for the output files')

args = parser.parse_args()

skip_mesh_manipulation = args.skip_mesh_manipulation
skip_abaqus = args.skip_abaqus
skip_dat_analyse = args.skip_dat_analyse

LVEndoAngle = args.endo
LVEpiAngle = args.epi
LayerNum = args.layer
process_ground_path = args.path
process_ground_path = process_ground_path + ('/' if process_ground_path[-1] != '/' else '')
geom = args.geom

# suffix = args.suffix
suffix = str(LVEndoAngle) + '_' + str(LVEpiAngle) + '_' + str(LayerNum) + '_' + str(geom) + '_'

# make sure the process_ground_path exists
if not os.path.exists(process_ground_path):
    os.makedirs(process_ground_path) 

if geom == -1:
    surf_file = args.surf
    set_file = args.sets
    node_file = args.node
    elem_file = args.elem 
else:
    surf_file = "../mesh/mesh_files/"+ str(geom)+"_surf.txt" 
    set_file ="../mesh/mesh_files/"+ str(geom)+"_set.txt"   
    node_file ="../mesh/mesh_files/"+ str(geom)+"_node.txt"  
    elem_file ="../mesh/mesh_files/"+ str(geom)+"_elem.txt"  


if not skip_mesh_manipulation:
    print( '.....Mesh manipulation begins')
    #############  epi_elems
    epi_elems = []
    with open(surf_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "epi" in line:
                line = sfile.readline()
                while "*" not in line:
                    epi_elems.append(line.strip().split(','))
                    line = sfile.readline()
                break
    epi_elems = [item for sublist in epi_elems for item in sublist]
    epi_elems = [item for item in epi_elems if item != '']
    del epi_elems[1::2]
    epi_elems = [int(item) for item in epi_elems]

    #############  RVendo_elems
    RVendo_elems = []
    with open(surf_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "RV" in line:
                line = sfile.readline()
                while "*" not in line:
                    RVendo_elems.append(line.strip().split(','))
                    line = sfile.readline()
                break
    RVendo_elems = [item for sublist in RVendo_elems for item in sublist]
    RVendo_elems = [item for item in RVendo_elems if item != '']
    del RVendo_elems[1::2]
    RVendo_elems = [int(item) for item in RVendo_elems]

    #############  LVendo_elems
    LVendo_elems = []
    with open(surf_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "LV" in line:
                line = sfile.readline()
                while line != '':
                    LVendo_elems.append(line.strip().split(','))
                    line = sfile.readline()
                break
    LVendo_elems = [item for sublist in LVendo_elems for item in sublist]
    LVendo_elems = [item for item in LVendo_elems if item != '']
    del LVendo_elems[1::2]
    LVendo_elems = [int(item) for item in LVendo_elems]

    #############  epi_nodes
    epi_nodes = []
    with open(set_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "epi" in line:
                line = sfile.readline()
                while "*" not in line:
                    epi_nodes.append(line.strip().split(','))
                    line = sfile.readline()
                break
    epi_nodes = [item for sublist in epi_nodes for item in sublist]
    epi_nodes = [item for item in epi_nodes if item != '']
    epi_nodes = [int(item) for item in epi_nodes]

    #############  RVendo_nodes
    RVendo_nodes = []
    with open(set_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "RV" in line:
                line = sfile.readline()
                while "*" not in line:
                    RVendo_nodes.append(line.strip().split(','))
                    line = sfile.readline()
                break
    RVendo_nodes = [item for sublist in RVendo_nodes for item in sublist]
    RVendo_nodes = [item for item in RVendo_nodes if item != '']
    RVendo_nodes = [int(item) for item in RVendo_nodes]

    #############  LVendo_nodes 
    LVendo_nodes = []
    with open(set_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "LV" in line:
                line = sfile.readline()
                while line != '':
                    LVendo_nodes.append(line.strip().split(','))
                    line = sfile.readline()
                break
    LVendo_nodes = [item for sublist in LVendo_nodes for item in sublist]
    LVendo_nodes = [item for item in LVendo_nodes if item != '']
    LVendo_nodes = [int(item) for item in LVendo_nodes]

    with open(process_ground_path + 'RVSets.txt', 'w+') as file:
        file.write(u'*Nset, nset=Set-RV_ENDO \n')
        for node in RVendo_nodes:
            file.write(u"%i\n" % node)
        file.write(u'*Elset, elset=Set-RV_ENDO \n')
        for elem in RVendo_elems:
            file.write(u"%i\n" % elem)

    with open(process_ground_path + 'LVSets.txt', 'w+') as file:
        file.write(u'*Nset, nset=Set-LV_ENDO \n')
        for node in LVendo_nodes:
            file.write(u"%i\n" % node)
        file.write(u'*Elset, elset=Set-LV_ENDO \n')
        for elem in LVendo_elems:
            file.write(u"%i\n" % elem)
            
    with open(process_ground_path + 'EPISets.txt', 'w+') as file:
        file.write(u'*Nset, nset=Set-EPI \n')
        for node in epi_nodes:
            file.write(u"%i\n" % node)
        file.write(u'*Elset, elset=Set-EPI \n')
        for elem in epi_elems:
            file.write(u"%i\n" % elem)
            
    with open(node_file, 'r') as file:
        lines=file.readlines()
    Nnodes= len(lines)
    with open (process_ground_path + 'AllNodes.txt', 'w+' ) as file:
        for line in lines:
            file.write(line)

    with open(elem_file, 'r') as file:
        lines=file.readlines()
    Nelems= len(lines)
    with open (process_ground_path + 'AllElements.txt', 'w+' ) as file:
        for line in lines:
            file.write(line)
            
    all_elems_set = [1, Nelems, 1]
    all_nodes_set = [1, Nnodes, 1]

    with open(process_ground_path + 'BVSets.txt', 'w+') as file:
        file.write(u'*Nset, nset=Set-BVCell, generate \n')
        file.write(u", ".join(str(x) for x in all_nodes_set))
        file.write(u'\n*Elset, elset=Set-BVCell, generate \n')
        file.write(u", ".join(str(x) for x in all_elems_set))

    ############ top_nodes (for top centroid calculation)
    top_nodes = []
    with open(set_file) as sfile:
        line = 'start'
        while line != '':
            line = sfile.readline()
            if "top" in line:
                line = sfile.readline()
                while "*" not in line:
                    top_nodes.append(line.strip().split(','))
                    line = sfile.readline()
                break
    top_nodes = [item for sublist in top_nodes for item in sublist]
    top_nodes = [item for item in top_nodes if item != '']
    top_nodes = [int(item) for item in top_nodes]

    nodes = np.loadtxt(process_ground_path + 'AllNodes.txt', delimiter=',')
    top_nodes_position = nodes [[i-1 for i in top_nodes],:]

    top_nodes_centroid=np.array([sum(top_nodes_position[:,1]), sum(top_nodes_position[:,2]), sum(top_nodes_position[:,3])])/len(top_nodes_position[:,1])
    np.savetxt(process_ground_path + 'top_nodes_centroid.txt', top_nodes_centroid, delimiter=',')

    print( '....Mesh manipulation Ends')



print('********Controller begins*********')

for name, content in INP_FILES.items():
    with open(process_ground_path + u'/' + name + u'.inp', u'w') as f:
        # convert to unicode
        content = content.decode(u'utf-8')
        f.write(content)
        
print('Files copied')
sys.stdout.flush()

os.chdir(process_ground_path)

for _cmnd in [u'abaqus job=HT_Analysis_BV_b', u'abaqus job=HT_Analysis_BV_c', u'abaqus job=HT_Analysis_BV_d']:
    _cmnd = _cmnd + u' interactive'
    print('>> ' + _cmnd)
    sys.stdout.flush()
    subprocess.call(_cmnd , shell=True)

print('Abaqus call completed')
sys.stdout.flush()


Nodes = np.loadtxt(fname=u'AllNodes.txt', delimiter=u',')
Elements = np.loadtxt(fname=u'AllElements.txt', delimiter=u',')


NNode = Nodes.shape[0]
NElem = Elements.shape[0]



Temps_b = process_dat_file(NElem, dat_file_address=u'./HT_Analysis_BV_b.dat')
Temps_c = process_dat_file(NElem, dat_file_address=u'./HT_Analysis_BV_c.dat')
Temps_d = process_dat_file(NElem, dat_file_address=u'./HT_Analysis_BV_d.dat')


Temps_cd = np.empty((NElem, 2))
print('@@ Calculating difference between BV_c and BV_d...')
sys.stdout.flush()
for i in xrange(1, NElem + 1):
    Temps_cd[i-1, 0] = i
    Temps_cd[i-1, 1] = -abs(Temps_c[Temps_c[:,0] == i][0][1] - Temps_d[Temps_d[:,0] == i][0][1])


Temps = np.empty((NElem, 2))
print('@@ Calculating average of BV_b and BV_cd...')
sys.stdout.flush()
for i in xrange(1, NElem + 1):
    Temps[i-1, 0] = i
    Temps[i-1, 1] = ((Temps_b[Temps_b[:,0] == i][0][1] + Temps_cd[Temps_cd[:,0] == i][0][1]) + 10) / 2



print('Layering Elements...')
sys.stdout.flush()

LayerDict = {}
for i in xrange(1, LayerNum + 1):
    LayerDict[u'Layer%s' % i] = []

for i in xrange(NElem):
    for j in xrange(1, LayerNum + 1):
        if Temps[i][1] <= (j * (10/LayerNum)):
            LayerDict[u'Layer%s' % j].append(Temps[i][0])
            break
        elif Temps[i][1] > 10:
            LayerDict[u'Layer%s' % LayerNum].append(Temps[i][0])
            break

Tetha = np.empty((NElem, 2), dtype=float)
Tetha[:, 0] = Elements[:, 0]

print('Calculating Angles...')
sys.stdout.flush()

for i in Elements[:, 0]:
    for j in xrange(1, LayerNum + 1):
        if i in LayerDict[u'Layer%s' % j]:
            Angle = LVEndoAngle + (j - 1) * ((LVEpiAngle - LVEndoAngle) / (LayerNum - 1))
            Tetha[np.where(Tetha[:, 0] == i)] = [i, Angle]
            break

print('Saving Angles file...')
np.savetxt(u'Angles.txt', Tetha)


print('******** Controller Complete **********')
sys.stdout.flush()

angle2orient_template_path = os.path.join(TEMPLATE_DIR, 'angle2orient.m.j2')
with open(angle2orient_template_path) as f:
    MATLAB_FILE_angle2orient = f.read()

MATLAB_FILE_angle2orient = jinja2.Template(MATLAB_FILE_angle2orient).render(**fiber_configs.data)

with open('angle2orient.m', 'w') as f:
    f.write(MATLAB_FILE_angle2orient.decode('utf-8'))

print('******** Fiber Synthesizer Begins **********')
matlab_command = ["matlab","-nodisplay","-nosplash","-nodesktop","-r","angle2orient('"+str(suffix)+"');exit;"]
print(">>",  ' '.join(matlab_command))
subprocess.call(matlab_command)