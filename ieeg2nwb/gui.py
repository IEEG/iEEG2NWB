
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout, QLineEdit, QComboBox,
                             QFileDialog, QFormLayout, QDialogButtonBox,
                             QMainWindow, QDialog, QPlainTextEdit, QGroupBox,
                             QTabWidget, QCheckBox, QDockWidget, QGridLayout,
                             QScrollArea, QTreeWidget, QTreeWidgetItem,
                             QAction, QToolBar,QMenuBar, QMenu)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
import sys
import tdt
import yaml
from .utils import load_nwb_settings

def isempty(variable):
    if isinstance(variable,dict):
        return len(variable.keys()) == 0
    else:
        return (variable==None) or (len(variable)==0)

def parse_filename(fname):
    outdict = {}
    pdir = os.path.dirname(fname)
    if not isempty(os.path.dirname(pdir)):
        outdict['folder'] = pdir

    fbase = os.path.basename(fname).split('.')
    if len(fbase) == 2:
        outdict['ext'] = fbase[1]

    fbase = fbase[0]
    fparts = fbase.split('_')
    if fparts[-1].count('-') == 0:
        outdict['ftype'] = fparts[-1]
        fparts.remove(fparts[-1])

    outdict.update( dict( part.split('-') for part in fparts) )
    return outdict

class GUI(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self._defaults = load_nwb_settings()

        # Set general window
        self._data = {}
        self.resize(QSize(900,600))
        self.root = QWidget()
        root_lay = QVBoxLayout()
        self.setWindowTitle('iEEG2NWB GUI')

        # Make tabs
        self.tab = QTabWidget()
        self.createRequiredTab()
        self.createSubjectTab()
        self.createMetaTab()
        self.createAnalogTab()
        self.createDigitalTab()
        self.createOutputTab()

        # Bottom Buttons
        bttmBttnsWidget = QWidget()
        bttmBttns = QHBoxLayout()

        quitBttn = QPushButton("Quit")
        quitBttn.clicked.connect(QApplication.instance().quit)
        bttmBttns.addWidget(quitBttn)

        resetBttn = QPushButton("Reset")
        resetBttn.clicked.connect(self.reset)
        bttmBttns.addWidget(resetBttn)

        loadBttn = QPushButton("Load")
        loadBttn.clicked.connect(self.load_params)
        bttmBttns.addWidget(loadBttn)

        saveBttn = QPushButton("Save")
        saveBttn.clicked.connect(self.save_params)
        bttmBttns.addWidget(saveBttn)

        runBttn = QPushButton("Run")
        runBttn.clicked.connect(self.run)
        bttmBttns.addWidget(runBttn)

        bttmBttnsWidget.setLayout(bttmBttns)

        # Add Widgets
        root_lay.addWidget(self.tab)
        root_lay.addWidget(bttmBttnsWidget)
        self.root.setLayout(root_lay)
        self.setCentralWidget(self.root)

        # Add dock for tree
        self.dock = QDockWidget('Dockable',self)
        self._tree = StoreListDlg([])
        self.dock.setWidget(self._tree)
        self.dock.setFloating(False)
        self.dock.setWindowTitle('Raw Data Overview')
        self.dock.setMinimumSize(QSize(400,300))
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

    def createRequiredTab(self):
        raw_lay = QFormLayout()
        raw_lay.setFormAlignment(Qt.AlignLeft)
        raw_lay.setLabelAlignment(Qt.AlignLeft)
        raw_lay.setVerticalSpacing(2)

        block = EegReader('(ex. B1_Visloc)',tooltip='Raw data block')
        block.newfile.connect(self.editTree)
        raw_lay.addRow(QLabel('Raw Data'),block)
        self._data['block'] = block

        labelfile = DataEntry('(ex. NS001_correspondence.xlsx)',isFile=True,tooltip='Correspondence sheet')
        raw_lay.addRow(QLabel('Labelfile'), labelfile)
        self._data['labelfile'] = labelfile

        neurodataEntry = DataEntry('EEG1, EEG2, RSn1', isFile=False, islist=True, tooltip='Comma-separated list of stores that contain the neural data that pertain to the correspondence sheet (only for TDT data). Must all have the same fs')
        raw_lay.addRow(QLabel('Neurodata'), neurodataEntry)
        self._data['neurodata'] = neurodataEntry

        raw_widget = QWidget()
        raw_widget.setLayout(raw_lay)
        self.tab.addTab(raw_widget,'Required')

    def createSubjectTab(self):
        defaults = self._defaults['subject']
        subinfo_lay = QFormLayout()
        subinfo_lay.setFormAlignment(Qt.AlignLeft)
        subinfo_lay.setLabelAlignment(Qt.AlignLeft)
        subinfo_lay.setVerticalSpacing(2)
        subid = DataEntry('(ex. NS001)',tooltip='subject id, default is %s' % defaults['subject_id'])
        subid.editor.textChanged.connect(lambda: self._data['output']._data['sub'].setValue(subid.getValue()) )
        subinfo_lay.addRow('Subject ID',subid)
        self._data['subject_id'] = subid

        age = DataEntry('(ex. 24)', tooltip="integer of the subject's age, default is %s" % defaults['age'])
        subinfo_lay.addRow('Age',age)
        self._data['age'] = age

        sex = DataEntry(['U','M','F'],tooltip = 'participant sex')
        subinfo_lay.addRow('Sex',sex)
        self._data['sex'] = sex

        sub_desc = DataEntry('(ex. had a prior resection of MTL)', tooltip='any additional info about the participant, default is %s' % defaults['description'])
        subinfo_lay.addRow('Description',sub_desc)
        self._data['subject_description'] = sub_desc

        subinfo_widget = QWidget()
        subinfo_widget.setLayout(subinfo_lay)
        self.tab.addTab(subinfo_widget,'Subject')

    def createMetaTab(self):
        defaults = self._defaults['meta_data']
        meta_fields = {
            'session_id': {'field': 'Session ID', 'tooltip': 'session ID of the experiment','placeholder': '(ex. implant01 - Day 1)'},
            'notes': {'field': 'Notes', 'tooltip': 'any notes relevant to the block','placeholder': '(ex. participant wore reading glasses)'},
            'session_description': {'field': 'Session Description', 'tooltip': 'description of the experiment session including relevant blocks done before and after','placeholder': '(ex. Rest, Visloc Block 1, Vicloc Block 2)'},
            'experiment_description': {'field': 'Experiment Description', 'tooltip': 'description of the experiment','placeholder': '(ex. n-back with pictures including tools, faces and houses)'},
            'lab': {'field': 'Lab', 'tooltip': 'the lab(s) associated with this experiment, default is "Human Brain Mapping Lab"','placeholder': '(ex. Schroeder Lab (NKI), The Human Brain Mapping Laboratory at The Feinstein Institutes for Medical Research)'},
            #'identifier': {'field': 'Identifier', 'tooltip': 'any information to specifically identify this file','placeholder': ''},
            'data_collection': {'field': 'Data Collection', 'tooltip': 'info about data collection such as original block name and ref electrodes','placeholder': '(ex. raw block-B1_VisLoc; ref=RFx15, Gnd=RPc16)'}
            }

        meta_lay = QFormLayout()
        meta_lay.setFormAlignment(Qt.AlignLeft)
        meta_lay.setLabelAlignment(Qt.AlignLeft)
        meta_lay.setVerticalSpacing(5)
        for f in meta_fields.keys():
            placeHolderTxt = meta_fields[f]['placeholder']
            placeHolderTxt += ', default is "%s"' % defaults[f]
            entry = DataEntry(placeHolderTxt, tooltip=meta_fields[f]['tooltip'])
            meta_lay.addRow(meta_fields[f]['field'],entry)
            self._data[f] = entry

        meta_widget = QWidget()
        meta_widget.setLayout(meta_lay)
        self.tab.addTab(meta_widget,'Meta-Data')

    def createAnalogTab(self):
        self._data['analog'] = []
        ana_lay = QVBoxLayout()
        ana_lay.setAlignment(Qt.AlignLeft)

        ana1_dict = {
            'name': {'field': 'Name', 'tooltip': 'Name to store the data as', 'placeholder': '(ex. audio)'},
            'stores': {'field': 'Store', 'tooltip': 'TDT store/container or EDF channel in which the data is stored', 'placeholder': '(ex. Wav5)'},
            'chans': {'field': 'Channel Numbers', 'tooltip': 'the channels where the data is stored, only applicable for TDT data', 'placeholder': '(ex. 1,2)'},
            'description': {'field': 'Description', 'tooltip': 'description of the data', 'placeholder': '(ex. audio played from computer)'},
            'comments': {'field': 'Comments', 'tooltip': 'comments about the data', 'placeholder': '(ex. a small ping at the beginning from computer)'}
            }
        ana1 = AdditionalStore(ana1_dict,header='Analog #1')
        ana_lay.addWidget(ana1)
        self._data['analog'].append(ana1)

        ana2_dict = {
            'name': {'field': 'Name', 'tooltip': 'Name to store the data as', 'placeholder': '(ex. ekg)'},
            'stores': {'field': 'Store', 'tooltip': 'TDT store/container or EDF channel in which the data is stored', 'placeholder': '(ex. EEG2)'},
            'chans': {'field': 'Channel Numbers', 'tooltip': 'the channels where the data is stored, only applicable for TDT data', 'placeholder': '(ex. 15)'},
            'description': {'field': 'Description', 'tooltip': 'description of the data', 'placeholder': '(ex. EKG from participant)'},
            'comments': {'field': 'Comments', 'tooltip': 'comments about the data', 'placeholder': '(ex. ekg placed on chest)'}
            }

        ana3_dict = {
            'name': {'field': 'Name', 'tooltip': 'Name to store the data as', 'placeholder': '(ex. TTL)'},
            'stores': {'field': 'Store', 'tooltip': 'TDT store/container or EDF channel in which the data is stored', 'placeholder': '(ex. DC2)'},
            'chans': {'field': 'Channel Numbers', 'tooltip': 'the channels where the data is stored, only applicable for TDT data', 'placeholder': ''},
            'description': {'field': 'Description', 'tooltip': 'description of the data', 'placeholder': '(ex. TTL pulses recorded in analog channel)'},
            'comments': {'field': 'Comments', 'tooltip': 'comments about the data', 'placeholder': ''}
            }


        ana2 = AdditionalStore(ana2_dict,header='Analog #2')
        self._data['analog'].append(ana2)
        ana3 = AdditionalStore(ana3_dict,header='Analog #3')
        self._data['analog'].append(ana3)
        ana4 = AdditionalStore(ana1_dict.copy(),header='Analog #4')
        self._data['analog'].append(ana4)
        ana5 = AdditionalStore(ana1_dict.copy(),header='Analog #5')
        self._data['analog'].append(ana5)
        ana6 = AdditionalStore(ana1_dict.copy(),header='Analog #6')
        self._data['analog'].append(ana6)

        ana_lay.addWidget(ana2)
        ana_lay.addWidget(ana3)
        ana_lay.addWidget(ana4)
        ana_lay.addWidget(ana5)
        ana_lay.addWidget(ana6)
        ana_widget = QWidget()
        ana_widget.setLayout(ana_lay)
        ana_scroll = QScrollArea()
        ana_scroll.setWidget(ana_widget)
        self.tab.addTab(ana_scroll,'Analog Stores')

    def createDigitalTab(self):
        # Digital Stores to add
        self._data['digital'] = []
        dig_lay = QVBoxLayout()
        dig_lay.setAlignment(Qt.AlignLeft)

        dig1_dict = {
            'name': {'field': 'Name', 'tooltip': 'Name to store the data as', 'placeholder': '(ex. TTL)'},
            'stores': {'field': 'Store', 'tooltip': 'list of TDT store where data is stored', 'placeholder': '(ex. PtC2,PtC4,PtC6)'},
            'description': {'field': 'Description', 'tooltip': 'description of the data', 'placeholder': '(ex. audio played from computer)'},
            'comments': {'field': 'Comments', 'tooltip': 'comments about the data', 'placeholder': '(ex. a small ping at the beginning from computer)'}
            }
        dig1 = AdditionalStore(dig1_dict,header='Digital #1',is_analog=False)
        dig_lay.addWidget(dig1)
        self._data['digital'].append(dig1)

        dig2_dict = {
            'name': {'field': 'Name', 'tooltip': 'Name to store the data as', 'placeholder': '(ex. TTL)'},
            'stores': {'field': 'Store', 'tooltip': 'list of TDT store where data is stored', 'placeholder': '(ex. PtC2,PtC4,PtC6)'},
            'description': {'field': 'Description', 'tooltip': 'description of the data', 'placeholder': '(ex. audio played from computer)'},
            'comments': {'field': 'Comments', 'tooltip': 'comments about the data', 'placeholder': '(ex. a small ping at the beginning from computer)'}
            }
        dig2 = AdditionalStore(dig1_dict,header='Digital #2',is_analog=False)
        dig_lay.addWidget(dig2)
        self._data['digital'].append(dig2)

        dig_widget = QWidget()
        dig_widget.setLayout(dig_lay)
        dig_scroll = QScrollArea()
        dig_scroll.setWidget(dig_widget)
        self.tab.addTab(dig_scroll,'Digital Stores')

    def createOutputTab(self):

        outputWidget = OutputWidget()
        self._data['output'] = outputWidget
        self.tab.addTab(outputWidget,'Output')

    def getData(self):
        output = {}
        for k in self._data.keys():
            if isinstance(self._data[k], list):
                output[k] = []
                ii = 0
                for ii in range( len(self._data[k]) ):
                    data = self._data[k][ii].getValue()
                    if not isempty(data):
                        output[k].append(data)

                if isempty(output[k]):
                    output.pop(k)

            else:
                data = self._data[k].getValue()
                if not isempty(data):
                    output[k] = data

        output['create_path'] = self._data['output'].createPath()
        return output

    def save_params(self):
        data = self.getData()
        print(data)
        fname = QFileDialog.getSaveFileName(self,'Save File')
        if len(fname) > 0:
            with open(fname[0],'w') as f:
                yaml.dump(data,f)

    def load_params(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            fname = dlg.selectedFiles()
            with open(fname[0],'r') as file:
                params = yaml.load(file, Loader=yaml.FullLoader)

            self.set_params(params)


    def set_params(self,params):
        self.reset()
        for k in params.keys():
            if k in self._data.keys():
                if isinstance(params[k],list) and not isinstance(self._data[k], DataEntry):
                    for ii in range( len(params[k]) ):
                        self._data[k][ii].setValue(params[k][ii])
                else:
                    self._data[k].setValue(params[k])
            else:
                print('Unknown field in params: %s' % k)


    def reset(self):
        for k in self._data.keys():
            if isinstance(self._data[k],list):
                for ii in range( len( self._data[k]) ):
                    self._data[k][ii].reset()
            else:
                self._data[k].reset()

    def run(self):
        from .iEEG2NWB import iEEG2NWB
        params = self.getData()
        inwb = iEEG2NWB()
        inwb.parse_params(params)

    def synthOutput(self):
        outputInfo = self._outputInfo
        required = ['subid','session','task']
        others = ['acq','run']
        for req in required:
            if isempty(outputInfo[req].getValue()):
                print('Missing required input: %s' % req)

        base = 'sub-%s_ses-%s_task-%s' % (outputInfo['subid'].getValue(),outputInfo['session'].getValue(), outputInfo['task'].getValue())

        if not isempty(outputInfo['acq'].getValue()):
            base += '_acq-%s' % outputInfo['acq'].getValue()

        if not isempty(outputInfo['run'].getValue()):
            if not outputInfo['run'].isnumeric():
                print('Error! "run" must be an integer')
            else:
                base += '_run-%d' % int(outputInfo['run'].getValue())

        base += '_ieeg.nwb'

        if (not isempty(outputInfo['folder'].getValue())) and (not os.path.isdir(outputInfo['folder'].getValue())):
            print('Error! Output folder must exist is being specified')
        else:
             self._data['output'].setValue( os.path.join(outputInfo['folder'].getValue(),base) )

        self._data['subject_id'].setValue(outputInfo['subid'].getValue())

    def editTree(self):
        self._tree.root.clear()
        self._tree.setOverview(self._data['block'].storeList)


    def createMenu(self):

        menu = self.menuBar()
        menu.setNativeMenuBar(False)

        # File
        fileMenu = menu.addMenu('File')
        loadMenu = fileMenu.addMenu('Load')
        loadParamsAction = QAction('Params file',self)
        loadParamsAction.triggered.connect(self.load_params)
        loadMenu.addAction(loadParamsAction)
        taskAction = QAction('Task',self)
        taskAction.triggered.connect(self.taskWindow.show)
        loadMenu.addAction(taskAction)
        saveAction = QAction('Save',self)
        saveAction.triggered.connect(self.save_params)
        fileMenu.addAction(saveAction)

        # Edit
        editMenu = menu.addMenu('Edit')
        resetAction = QAction('Reset',self)
        resetAction.triggered.connect(self.reset)
        editMenu.addAction(resetAction)

class DataEntry(QWidget):

    def __init__(self,values,parent=None,isFile = False, tooltip='',islist=False):
        QWidget.__init__(self,parent=parent)

        self.lay = QHBoxLayout(self)
        self.islist = islist

        if isinstance(values,list):
            self.addComboBox(values)
        elif isinstance(values,str):
            self.addLineEditor(placeholder=values,isFile=isFile)
        else:
            print('Unknown data type')

        self.addToolTip(tooltip)

    def addLineEditor(self,placeholder='',isFile=False):
        editor = QLineEdit()
        editor.setMinimumWidth(500)
        editor.setPlaceholderText(placeholder)
        self.lay.addWidget(editor)
        self.editor = editor

        # If the line is supposed to be a file
        if isFile:
            bttn = QPushButton('...')
            self.lay.addWidget(bttn)
            bttn.clicked.connect(self.file_search)
            self.setAcceptDrops(True)
            self.lay.addWidget(bttn)

    def addComboBox(self,values):
        editor = QComboBox()
        editor.addItems(values)
        editor.setMinimumWidth(250)
        self.lay.addWidget(editor)
        self.editor = editor

    def addToolTip(self,tooltip):
        # Question mark tooltip
        helper = QPushButton('?')
        helper.setToolTip(tooltip)
        self.lay.addWidget(helper)

    def getValue(self):
        if isinstance(self.editor,QLineEdit):
            txt = self.editor.text()
        elif isinstance(self.editor,QComboBox):
            txt = self.editor.currentText()
        if self.islist:
            txt = [ii.rstrip().lstrip() for ii in txt.split(',')]

        return txt

    def dragEnterEvent(self, e):
        e.accept()
        #print(e)

    def dropEvent(self,e):
        #print(e.mimeData())
        self.editor.setText( e.mimeData().text().replace('file://', '' ) )

    def file_search(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            self.editor.setText(filenames[0])
            return 1
        else:
            return 0

    def reset(self):
        if isinstance(self.editor,QLineEdit):
            self.editor.clear()
        elif isinstance(self.editor,QComboBox):
            self.editor.setCurrentIndex(0)

    def setValue(self,value):

        if self.islist and isinstance(value, list):
            value = ','.join(value)

        if isinstance(self.editor,QLineEdit):
            self.editor.setText( str(value) )
        elif isinstance(self.editor,QComboBox):
            self.editor.setCurrentText(value.upper())

class EegReader(DataEntry):

    newfile = pyqtSignal()

    def __init__(self,values,tooltip=''):
        DataEntry.__init__(self,values,isFile = True, tooltip='')

    def file_search(self):
        fileIsReal = DataEntry.file_search(self)
        if fileIsReal:
            self.readHeader()
            self.newfile.emit()
        else:
            print('--->Can not find file')

    def readHeader(self):
        fname = self.editor.text()
        if not os.path.exists(fname):
            stores = [{'name': 'File or Directory Not Found!', 'dtype': '', 'chans': '', 'fs': '', 'samples': ''}]
        else:
            hdr = tdt.read_block(fname,headers=True)
            stores = []
            for s in hdr.stores.keys():
                thisStore = hdr.stores[s]
                storeTypes2Use = ['epocs','streams']
                if thisStore['type_str'] == 'streams':
                    storeType = 'analog'
                    nchans = max(thisStore.chan)
                    fs = int(thisStore.fs)
                    nsamples = 'Unknown'
                elif thisStore['type_str'] == 'epocs':
                    storeType = 'digital'
                    nchans = 1
                    fs = 'NA'
                    nsamples = thisStore.onset.size
                else:
                    continue

                stores.append({
                    'name': s,
                    'dtype': storeType,
                    'chans': nchans,
                    'fs': fs,
                    'samples': nsamples})

        self.storeList = stores
        return stores

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
            self.readHeader()
            self.newfile.emit()

class AdditionalStore(QGroupBox):

    def __init__(self,info,parent=None,header='Analog',is_analog=True):
        QGroupBox.__init__(self)

        self._data = {}

        lay = QFormLayout()
        lay.setLabelAlignment(Qt.AlignLeft)

        name = DataEntry(info['name']['placeholder'],tooltip=info['name']['tooltip'])
        self._data['name'] = name
        lay.addRow('Name',name)

        store = DataEntry(info['stores']['placeholder'],tooltip=info['stores']['tooltip'])
        self._data['stores'] = store
        lay.addRow('Stores',store)

        if is_analog:
            externalizeCheckBoxText = 'Write to external .wav file.\nDo this for microphone.\nMicrophone with patient voice is PHI'
            chans = DataEntry(info['chans']['placeholder'],tooltip=info['chans']['tooltip'])
            self._data['channels'] = chans
            lay.addRow('Channel Numbers',chans)
        else:
            externalizeCheckBoxText = 'Write to .csv\nTTLs only needed\nfor event extration,\nnot permanent storage'

        desc = DataEntry(info['description']['placeholder'],tooltip=info['description']['tooltip'])
        self._data['description'] = desc
        lay.addRow('Description',desc)

        comments = DataEntry(info['comments']['placeholder'],tooltip=info['comments']['tooltip'])
        self._data['comments'] = comments
        lay.addRow('Comments',comments)

        # Checkbox for option to write data to external file
        self.externalizeCheckBox = QCheckBox()
        self.externalizeCheckBox.setChecked(False)
        externalLay = QHBoxLayout()
        externalWidget = QWidget()
        externalWidget.setLayout(externalLay)
        externalLay.addWidget(QLabel(externalizeCheckBoxText))
        externalLay.addWidget(self.externalizeCheckBox)
        lay.addWidget(externalWidget)

        self.setLayout(lay)
        self.setTitle(header)

    def reset(self):
        for k in self._data.keys():
            self._data[k].reset()

        self.externalizeCheckBox.setChecked(False)

    def getValue(self):
        data = {}
        for k in self._data.keys():
            val = self._data[k].getValue()
            if not isempty(val):
                if k == 'stores' or k == 'store':
                    data['stores'] = val.split(',')
                elif k == 'channels':
                    data['channels'] = val.split(',')
                else:
                    data[k] = val
        if len(data.keys()) > 0:
            data['externalize'] = self.externalizeCheckBox.isChecked()
        return data

    def setValue(self,values):
        self.reset()
        if 'store' in values.keys():
            values['stores'] = values['store']
        for k in values.keys():
            if k == 'externalize':
                doExternalize = str(values[k]).upper() in ['1','TRUE','YES','Y']
                self.externalizeCheckBox.setChecked(doExternalize)
            if k in self._data.keys():
                if isinstance(values[k],list):
                    tmp = [str(x) for x in values[k]]
                    #self._data[k].setValue(','.join(values[k]))
                    self._data[k].setValue(','.join(tmp))
                else:
                    self._data[k].setValue(values[k])


class StoreListDlg(QWidget):

    def __init__(self,storeList,showAll = True):
        QWidget.__init__(self)

        self.__sample = [{'name': 'Wav5', 'chans': 4, 'fs': 24000, 'samples': 1000,'dtype': 'analog'},
                  {'name': 'EEG1', 'chans': 100, 'fs': 1500, 'samples': 1000,'dtype': 'digital'}]

        self.resize(610,210)
        self.root = QTreeWidget(self)
        self.root.setColumnCount(5)

        self.root.setHeaderLabels(['Name','fs','samples','nchans','type'])
        self.root.resize(600, 200)
        self.root.setColumnWidth(0,90)
        self.root.setColumnWidth(1,80)
        self.root.setColumnWidth(2,80)
        self.root.setColumnWidth(3,80)
        self.root.setColumnWidth(4,80)


    def setAllChans(self,storeList):
        self.root.setHeaderLabels(['Name','fs','samples'])
        self.root.resize(600, 200)
        self.root.setColumnWidth(0,150)
        self.root.setColumnWidth(1,80)
        self.root.setColumnWidth(2,80)
        stores = {}

        for s in storeList:
            sname = s['name']
            fs = str( s['fs'] )
            nchans = s['chans']
            samples = str( s['samples'] )
            stores[sname] = QTreeWidgetItem([sname,fs,samples])

            for ii in range(nchans):
                chan_name = sname + '-' + str(ii+1).zfill(3)
                storeChn = QTreeWidgetItem([chan_name,fs,samples])
                stores[sname].addChild(storeChn)

            self.root.addTopLevelItem(stores[sname])

    def setOverview(self,storeList):
        self.root.clear()
        stores = {}
        for s in storeList:
            sname = s['name']
            fs = str( s['fs'] )
            nchans = str( s['chans'] )
            samples = str( s['samples'] )
            storeType = s['dtype']
            stores[sname] = QTreeWidgetItem([sname,fs,samples,nchans,storeType])
            self.root.addTopLevelItem(stores[sname])


class OutputWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._data = {}

        lay = QFormLayout()
        lay.setFormAlignment(Qt.AlignLeft)
        lay.setLabelAlignment(Qt.AlignLeft)
        lay.setVerticalSpacing(2)

        outDict = {
            'folder': ['Directory','/home/data/sub-NS001/ses-implant01/ieeg', 'parent directory to store the NWB file'],
            'sub': ['Subject ID*','autofilled','autofilled'],
            'ses': ['Session*', '(ex. implant01)','session id such as "implant01"'],
            'task': ['Task*','(ex. Visloc)','shorthand task name'],
            'acq': ['Acq', '(ex. classic1)','any specific variation or difference from typical experiment, use if an experiment has variations'],
            'run': ['Run','(ex. 01)','order number if experiment repeated']}

        for k in outDict.keys():
            label = outDict[k][0]
            placeHolder = outDict[k][1]
            helpInfo = outDict[k][2]
            self._data[k] = DataEntry(placeHolder,isFile=(k=='folder'),tooltip=helpInfo)
            lay.addRow(label,self._data[k])

        # Push button to display output and the output field
        applyBttn = QPushButton('Apply')
        applyBttn.clicked.connect(self.synthOutput)
        #lay.addWidget(applyBttn)

        # Checkbox1
        self.bidsPathCheckBox = QCheckBox()
        self.bidsPathCheckBox.setChecked(False)

        # Checkbox2
        self.createPathCheckBox = QCheckBox()
        self.createPathCheckBox.setChecked(False)

        # Output options
        outputBttnBox = QWidget()
        outputBttnBoxLay = QHBoxLayout()
        outputBttnBox.setLayout(outputBttnBoxLay)
        outputBttnBoxLay.addWidget(applyBttn)
        outputBttnBoxLay.addSpacing(30)
        outputBttnBoxLay.addWidget(QLabel('Create full BIDS path'))
        outputBttnBoxLay.addWidget(self.bidsPathCheckBox)
        outputBttnBoxLay.addSpacing(30)
        outputBttnBoxLay.addWidget(QLabel('Create missing directories'))
        outputBttnBoxLay.addWidget(self.createPathCheckBox)
        lay.addWidget(outputBttnBox)

        output = DataEntry('(ex. sub-NS001_ses-implant01_task-visloc_ieeg)',isFile=True,tooltip='Output file')
        outputLabel = QLabel('Output')

        self._data['output'] = output
        lay.addRow('Output', output)
        self.setLayout(lay)

    def synthOutput(self):
        outputInfo = self._data
        required = ['sub','ses','task']
        others = ['acq','run']
        for req in required:
            if isempty(outputInfo[req].getValue()):
                print('Missing required input: %s' % req)

        base = 'sub-%s_ses-%s_task-%s' % (outputInfo['sub'].getValue(),outputInfo['ses'].getValue(), outputInfo['task'].getValue())

        if not isempty(outputInfo['acq'].getValue()):
            base += '_acq-%s' % outputInfo['acq'].getValue()

        if not isempty(outputInfo['run'].getValue()):
            if not outputInfo['run'].getValue().isnumeric():
                print('Error! "run" must be an integer')
                sys.exit()
            else:
                base += '_run-%02d' % int(outputInfo['run'].getValue())

        base += '_ieeg.nwb'

        if (not isempty(outputInfo['folder'].getValue())) and (not os.path.isdir(outputInfo['folder'].getValue())) and (self.createPathCheckBox.isChecked() == False):
            print('Error! Output folder must exist if being specified OR set "Create Missing Directories" to True')
        else:
            if self.bidsPathCheckBox.isChecked():
                bidsPath = os.path.join('sub-' + outputInfo['sub'].getValue(), 'ses-' + outputInfo['ses'].getValue(), 'ieeg')
                self._data['output'].setValue( os.path.join(outputInfo['folder'].getValue(),bidsPath,base) )
            else:
                self._data['output'].setValue( os.path.join(outputInfo['folder'].getValue(),base) )

    def getValue(self,field='output'):
        return self._data[field].getValue()

    def setValue(self,value):
        self.reset()
        fparts = parse_filename(value)
        for k in fparts.keys():
            if k in self._data.keys():
                self._data[k].setValue(fparts[k])

        self._data['output'].setValue(value)

    def reset(self):
        for k in self._data.keys():
            if isinstance(self._data[k],list):
                for ii in range( len( self._data[k]) ):
                    self._data[k][ii].reset()
            else:
                self._data[k].reset()

    def createPath(self):
        return self.createPathCheckBox.isChecked()




if __name__ == '__main__':
    app = QApplication([])
    ex = GUI()
    #ex.show()
    sys.exit(app.exec_())
