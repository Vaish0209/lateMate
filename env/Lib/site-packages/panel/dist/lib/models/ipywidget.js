var _a;
import { InlineStyleSheet } from "@bokehjs/core/dom";
import { HTMLBox, HTMLBoxView } from "./layout";
const Jupyter = window.Jupyter;
export class IPyWidgetView extends HTMLBoxView {
    async lazy_initialize() {
        await super.lazy_initialize();
        let manager;
        if ((Jupyter != null) && (Jupyter.notebook != null))
            manager = Jupyter.notebook.kernel.widget_manager;
        else if (window.PyViz.widget_manager != null)
            manager = window.PyViz.widget_manager;
        else {
            console.warn("Panel IPyWidget model could not find a WidgetManager");
            return;
        }
        this.manager = manager;
        this.ipychildren = [];
        const { spec, state } = this.model.bundle;
        const models = await manager.set_state(state);
        const model = models.find((item) => item.model_id == spec.model_id);
        if (model != null) {
            const view = await this.manager.create_view(model, { el: this.el });
            this.ipyview = view;
            if (view.children_views) {
                for (const child of view.children_views.views)
                    this.ipychildren.push(await child);
            }
        }
    }
    _ipy_stylesheets() {
        const stylesheets = [];
        for (const child of document.head.children) {
            if (child instanceof HTMLStyleElement) {
                const raw_css = child.textContent;
                if (raw_css != null) {
                    const css = raw_css.replace(/:root/g, ":host");
                    stylesheets.push(new InlineStyleSheet(css));
                }
            }
        }
        return stylesheets;
    }
    stylesheets() {
        return [...super.stylesheets(), ...this._ipy_stylesheets()];
    }
    render() {
        super.render();
        if (this.ipyview != null) {
            this.shadow_el.appendChild(this.ipyview.el);
            this.ipyview.trigger('displayed', this.ipyview);
            for (const child of this.ipychildren)
                child.trigger('displayed', child);
            this.invalidate_layout();
        }
    }
}
IPyWidgetView.__name__ = "IPyWidgetView";
export class IPyWidget extends HTMLBox {
    constructor(attrs) {
        super(attrs);
    }
}
_a = IPyWidget;
IPyWidget.__name__ = "IPyWidget";
IPyWidget.__module__ = "panel.models.ipywidget";
(() => {
    _a.prototype.default_view = IPyWidgetView;
    _a.define(({ Any }) => ({
        bundle: [Any, {}],
    }));
})();
//# sourceMappingURL=ipywidget.js.map