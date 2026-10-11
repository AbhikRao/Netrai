classdef ReduceMeanLayer1022 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1022(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1022(cgInstance.Name);
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
        function this = ReduceMeanLayer1022(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_327'};
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

        function [x_model_backbone_327] = predict(this, x_model_backbone_321__)
            if isdlarray(x_model_backbone_321__)
                x_model_backbone_321_ = stripdims(x_model_backbone_321__);
            else
                x_model_backbone_321_ = x_model_backbone_321__;
            end
            x_model_backbone_321NumDims = 4;
            x_model_backbone_321 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_321_, [4 3 1 2], 4);

            [x_model_backbone_327__, x_model_backbone_327NumDims__] = ReduceMeanGraph1066(this, x_model_backbone_321, x_model_backbone_321NumDims, false);
            x_model_backbone_327_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_327__, [3 4 2 1], 4);

            x_model_backbone_327 = dlarray(single(x_model_backbone_327_), 'SSCB');
        end

        function [x_model_backbone_327, x_model_backbone_327NumDims1068] = ReduceMeanGraph1066(this, x_model_backbone_321, x_model_backbone_321NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1044 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1067, coder.const(x_model_backbone_321NumDims));
            xReduced1045 = mean(x_model_backbone_321, dims1044);
            x_model_backbone_327 = xReduced1045;
            x_model_backbone_327NumDims = coder.const(x_model_backbone_321NumDims);

            % Set graph output arguments
            x_model_backbone_327NumDims1068 = coder.const(x_model_backbone_327NumDims);

        end

    end

end