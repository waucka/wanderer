pub struct AppContext {
    should_quit: bool,
}

impl AppContext {
    pub fn new() -> Self {
	Self{
	    should_quit: false,
	}
    }

    pub fn signal_quit(&mut self) {
	self.should_quit = true;
    }

    pub fn quit_signaled(&self) -> bool {
	self.should_quit
    }
}

pub struct UniformTwiddler {
    label: String,
    value: f32,
    painting: Painting,
}

impl Default for UniformTwiddler {
    fn default() -> Self {
	Self{
	    label: "Hello World!".to_owned(),
	    value: 2.7,
	    painting: Default::default(),
	}
    }
}

impl UniformTwiddler {
    pub fn name(&self) -> &str {
	"Uniform Twiddler"
    }

    pub fn update(&mut self, ctx: &egui::CtxRef, app_ctx: &mut AppContext) {
	let UniformTwiddler {
	    label,
	    value,
	    painting,
	} = self;

	egui::SidePanel::left("side_panel", 200.0).show(ctx, |ui| {
	    ui.heading("Side Panel");

	    ui.horizontal(|ui| {
		ui.label("Write something: ");
		ui.text_edit_singleline(label);
	    });

	    ui.add(egui::Slider::f32(value, 0.0..=10.0).text("value"));
	    if ui.button("Increment").clicked {
		*value += 1.0;
	    }

	    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
		ui.add(
		    egui::Hyperlink::new("https://github.com/emilk/egui").text("powered by egui"),
		);
	    });
	});

	egui::TopPanel::top("top_panel").show(ctx, |ui| {
	    egui::menu::bar(ui, |ui| {
		egui::menu::menu(ui, "File", |ui| {
		    if ui.button("Quit").clicked {
			app_ctx.signal_quit();
		    }
		});
	    });
	});

	egui::CentralPanel::default().show(ctx, |ui| {
	    ui.heading("Uniform Twiddler");
	    egui::warn_if_debug_build(ui);

	    ui.separator();

	    ui.heading("Central Panel");
	    ui.label("The central panel is the region left after adding TopPanels and SidePanels");
	    ui.label("It is often a great place for big things, like drawings:");

	    ui.heading("Draw with your mouse:");
	    painting.ui_control(ui);
	    egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
		painting.ui_content(ui);
	    });
	});
    }
}

struct Painting {
    lines: Vec<Vec<egui::Vec2>>,
    stroke: egui::Stroke,
}

impl Default for Painting {
    fn default() -> Self {
	Self{
	    lines: Default::default(),
	    stroke: egui::Stroke::new(1.0, egui::Color32::LIGHT_BLUE),
	}
    }
}

impl Painting {
    pub fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
	ui.horizontal(|ui| {
	    self.stroke.ui(ui, "Stroke");
	    ui.separator();
	    if ui.button("Clear Painting").clicked {
		self.lines.clear()
	    }
	}).1
    }

    pub fn ui_content(&mut self, ui: &mut egui::Ui) -> egui::Response {
	let (response, painter) = ui.allocate_painter(ui.available_size_before_wrap_finite(), egui::Sense::drag());
	let rect = response.rect;

	if self.lines.is_empty() {
	    self.lines.push(vec![]);
	}

	let current_line = self.lines.last_mut().unwrap();

	if response.active {
	    if let Some(mouse_pos) = ui.input().mouse.pos {
		let canvas_pos = mouse_pos - rect.min;
		if current_line.last() != Some(&canvas_pos) {
		    current_line.push(canvas_pos);
		}
	    }
	} else if !current_line.is_empty() {
	    self.lines.push(vec![]);
	}

	for line in &self.lines {
	    if line.len() >= 2 {
		let points: Vec<egui::Pos2> = line.iter().map(|p| rect.min + *p).collect();
		painter.add(egui::PaintCmd::line(points, self.stroke));
	    }
	}

	response
    }
}
