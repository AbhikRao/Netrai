classdef ReduceMeanLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = netrai_fold0.coder.ReduceMeanLayer1003(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1003(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = ReduceMeanLayer1003(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone__46'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = netrai_fold0.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [x_model_backbone__46] = predict(this, x_model_backbone__40__)
            if isdlarray(x_model_backbone__40__)
                x_model_backbone__40_ = stripdims(x_model_backbone__40__);
            else
                x_model_backbone__40_ = x_model_backbone__40__;
            end
            x_model_backbone__40NumDims = 4;
            x_model_backbone__40 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone__40_, [4 3 1 2], 4);

            [x_model_backbone__46__, x_model_backbone__46NumDims__] = ReduceMeanGraph1009(this, x_model_backbone__40, x_model_backbone__40NumDims, false);
            x_model_backbone__46_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone__46__, [3 4 2 1], 4);

            x_model_backbone__46 = dlarray(single(x_model_backbone__46_), 'SSCB');
        end

        function [x_model_backbone__46, x_model_backbone__46NumDims1011] = ReduceMeanGraph1009(this, x_model_backbone__40, x_model_backbone__40NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1006 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1010, coder.const(x_model_backbone__40NumDims));
            xReduced1007 = mean(x_model_backbone__40, dims1006);
            x_model_backbone__46 = xReduced1007;
            x_model_backbone__46NumDims = coder.const(x_model_backbone__40NumDims);

            % Set graph output arguments
            x_model_backbone__46NumDims1011 = coder.const(x_model_backbone__46NumDims);

        end

    end

end