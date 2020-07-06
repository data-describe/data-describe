import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Initialization data for the data-describe-ext extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'data-describe-ext',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension data-describe-ext is activated!');
  }
};

export default extension;
