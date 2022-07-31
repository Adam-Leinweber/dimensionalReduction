import glob
import fileinput
import sys
from subprocess import call

#select param cards to generate processes for
param_cards = glob.glob('Cards/-72*param_card.dat')
#param_cards = glob.glob('Cards/*')
print(param_cards)
excluded = []

#select processes
processes = ['x1 n3', 'x1 n4', 'x2 n2', 'x2 n3', 'x2 n4', 'n2 n2', 'n2 n3', 'n2 n4', 'n3 n4', 'n3 n3', 'n4 n4', 'x1+ x1-', 'x2+ x2-'] 
# processes = ['x2+ x2-', 'n3 x2']

call(['mkdir', 'Generate/root_files'])
for param_card in param_cards:
    if param_card in excluded:
        continue

    call(['mkdir', 'Generate/'+param_card.replace('Cards/','').replace('_GAMBIT_param_card.dat','')])

    for process in processes:
        #open file
        with open('Cards/mg5_commands_jets.cmnd','r') as f:
            data = f.readlines()
        #edit lines
        for i in range(len(data)):
            if '@ 0' in data[i]:
                data[i] = 'generate p p > ' + process + ' @ 0\n'
            elif '@ 1' in data[i]:
                data[i] = 'add process p p > ' + process + ' j @ 1\n'
            elif '@ 2' in data[i]:
                data[i] = 'add process p p > ' + process + ' j j @ 2\n'
            elif 'output' in data[i]:
                data[i] = 'output Generate/'+param_card.replace('Cards/','').replace('_GAMBIT_param_card.dat','')+'/ppTO' + process.replace(' ','') + '_jets\n'
            elif 'param_card.dat' in data[i]:
                data[i] = param_card+'\n'
        #write file
        with open('Cards/mg5_commands_jets.cmnd','w') as f:
            f.writelines(data)

        #run madgraph with command file
        call(['./bin/mg5_aMC', 'Cards/mg5_commands_jets.cmnd'])
        #move root file to a dedicated directory
        call(['mv', 'Generate/'+param_card.replace('Cards/','').replace('_GAMBIT_param_card.dat','')+'/ppTO'+process.replace(' ','')+'_jets/Events/run_01/tag_1_delphes_events.root', 'Generate/root_files/'+param_card.replace('Cards/','').replace('_GAMBIT_param_card.dat','')+'_'+process.replace(' ','')+'_jets.root'])
